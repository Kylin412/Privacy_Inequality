# -*- coding: utf-8 -*-
"""An auction simulation."""
import argparse
import random
import time
import numpy as np
import warnings
import logging
import os
import sys
import pickle
import json
import shutil  # 用于删除目录


import matplotlib
matplotlib.use('Agg', force=True)  
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
import datetime 
import getpass 
import traceback
import networkx as nx 
import csv  # 添加CSV模块用于保存数据
import re  # 添加正则表达式模块用于日志清理
from typing import Optional

# 保存原始的stdout和stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

# 重设日志级别，允许更多的日志输出
logging.getLogger().setLevel(logging.INFO)

# 允许 agentscope 的日志
os.environ["AGENTSCOPE_LOG_LEVEL"] = "INFO"

from agents_complete import Consumer, Firm, Platform, Broadcaster
import agentscope

for logger_name in logging.root.manager.loggerDict:
    if logger_name.startswith('agentscope'):
        logging.getLogger(logger_name).setLevel(logging.INFO)
        logging.getLogger(logger_name).propagate = True


# 添加详细数据记录类
class DetailedDataRecorder:
    """记录详细数值特征到CSV文件"""
    
    def __init__(self, base_dir: str, exp_idx: int, firm_num: int, rational_tag: str):
        self.base_dir = base_dir
        self.exp_idx = exp_idx
        self.firm_num = firm_num
        self.rational_tag = rational_tag
        
        self.detailed_data_dir = os.path.join(base_dir, 'detailed_data')
        if not os.path.exists(self.detailed_data_dir):
            os.makedirs(self.detailed_data_dir)
                    
        self.csv_path = os.path.join(
            self.detailed_data_dir, 
            f'detailed_exp_{exp_idx + 1}_firms_{firm_num}_{rational_tag}.csv'
        )
        
        self._init_csv()
        
    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        fieldnames = [
            'Round', 'Agent_Type', 'Agent_Index', 
            # 消费者相关字段
            'Share_Decision', 'Share_Reason', 'Privacy_Cost', 'Valuations',
            'Searched_Firms', 'Purchase_Index', 'Total_Revenue', 'Search_Cost',
            'Decision_Sequence', 'Final_Reason',
            # 企业相关字段  
            'Price', 'Price_Reason', 'Share_Rate_Predicted', 'Revenue', 'Profit',
            'Personalized_Prices', 'Sales_Count',
            # 平台相关字段
            'Platform_Consumer_Surplus', 'Platform_Firm_Surplus', 'Platform_Share_Rate'
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
    def record_round_data(self, round_num: int, consumers: list, firms: list, platform, share_ratio: float):
        """记录一轮的详细数据"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Round', 'Agent_Type', 'Agent_Index', 
                'Share_Decision', 'Share_Reason', 'Privacy_Cost', 'Valuations',
                'Searched_Firms', 'Purchase_Index', 'Total_Revenue', 'Search_Cost',
                'Decision_Sequence', 'Final_Reason',
                'Price', 'Price_Reason', 'Share_Rate_Predicted', 'Revenue', 'Profit',
                'Personalized_Prices', 'Sales_Count',
                'Platform_Consumer_Surplus', 'Platform_Firm_Surplus', 'Platform_Share_Rate'
            ])
            
            for consumer in consumers:
                row = {
                    'Round': round_num + 1,
                    'Agent_Type': 'Consumer',
                    'Agent_Index': consumer.index,
                    'Share_Decision': consumer.share,
                    'Share_Reason': str(consumer.temp_memory.get('share_reason', '')),
                    'Privacy_Cost': consumer.privacy_cost,
                    'Valuations': str(consumer.valuations.tolist()),
                    'Searched_Firms': str(consumer.searched_firms),
                    'Purchase_Index': consumer.purchase_index,
                    'Total_Revenue': consumer.total_revenue,
                    'Search_Cost': consumer.total_search_cost,
                    'Decision_Sequence': str(consumer.temp_memory.get('decision_sequence', [])),
                    'Final_Reason': str(consumer.temp_memory.get('final_reason', '')),
                    # 企业字段留空
                    'Price': '', 'Price_Reason': '', 'Share_Rate_Predicted': '', 
                    'Revenue': '', 'Profit': '', 'Personalized_Prices': '', 'Sales_Count': '',
                    # 平台字段
                    'Platform_Consumer_Surplus': platform.consumer_surplus,
                    'Platform_Firm_Surplus': platform.firm_surplus,
                    'Platform_Share_Rate': share_ratio
                }
                writer.writerow(row)
                
            for firm in firms:
                row = {
                    'Round': round_num + 1,
                    'Agent_Type': 'Firm',
                    'Agent_Index': firm.index,
                    'Share_Decision': '', 'Share_Reason': '', 'Privacy_Cost': '', 'Valuations': '',
                    'Searched_Firms': '', 'Purchase_Index': '', 'Total_Revenue': '', 'Search_Cost': '',
                    'Decision_Sequence': '', 'Final_Reason': '',
                    'Price': firm.price,
                    'Price_Reason': str(firm.temp_memory.get('price_reason', '')),
                    'Share_Rate_Predicted': str(firm.temp_memory.get('share_rate_predicted', '')),
                    'Revenue': firm.revenue,
                    'Profit': firm.profit,
                    'Personalized_Prices': str(firm.personalized_prices),
                    'Sales_Count': firm.temp_memory.get('sale_num', 0),
                    'Platform_Consumer_Surplus': platform.consumer_surplus,
                    'Platform_Firm_Surplus': platform.firm_surplus,
                    'Platform_Share_Rate': share_ratio
                }
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Recommendation Simulation Arguments")
    parser.add_argument("--consumer-num", type=int, default=4, help="Number of consumers")
    parser.add_argument("--firm-num", type=int, default=5,
                        help="Number of firms (or starting number for multi-experiment runs)")
    parser.add_argument("--search-cost", type=float, default=0.02, help="Search cost for consumers")
    parser.add_argument(
        "--agent-type",
        choices=["random", "llm"],
        default="llm",
        help="Agent type for decision making",
    )
    parser.add_argument("--waiting-time", type=float, default=0.5,
                        help="Waiting time between steps (mainly for visual debugging)")
    parser.add_argument("--use-dist", action="store_true", help="Enable distributed mode")
    parser.add_argument("--visualize", action="store_true", help="Enable saving of visualization plots")
    parser.add_argument("--threads", type=int, default=5, help="Number of threads for parallel processing")
    parser.add_argument("--memory_truncate", type=int, default=3, help="Memory truncation length")
    parser.add_argument("--memory_distill", type=bool, default=False, help="Enable memory distillation")
    parser.add_argument("--model_config_name", type=str, default="gpt-config", help="Model configuration name to use")
    parser.add_argument("--clean_start", action="store_true", help="Delete previous experiment results and start fresh")
    parser.add_argument("--num-experiments", type=int, default=6,
                        help="Number of experiments to run (incrementing firm_num)")
    parser.add_argument("--num-rounds", type=int, default=4, help="Number of simulation rounds per experiment")
    parser.add_argument("--force-fresh", action="store_true",
                        help="Force this run to start from experiment 1, ignoring any checkpoint.")
    parser.add_argument("--pricing-mode", choices=["fixed", "adaptive", "perfect"], default="adaptive", help="Firm pricing mode")
    parser.add_argument("--firm-cost", type=float, default=0.0, help="Production cost per unit for firms")
    parser.add_argument("--top_k_consumers", type=int, default=3, help="Number of top consumers to broadcast (-1 for all)")
    parser.add_argument("--top_n_firms", type=int, default=3, help="Number of top firms to broadcast (-1 for all)")
    parser.add_argument("--distill_broadcast", action="store_true", help="Enable distillation for broadcast messages")
    parser.add_argument("--start_broadcast_round", type=int, default=2, 
                        help="从第几轮开始进行广播 (默认从第2轮开始)")
    parser.add_argument("--broadcast_history_window", type=int, default=1, 
                        help="广播时考虑的历史轮数窗口大小 (默认仅考虑前1轮)")

    # --- Network Structure Arguments ---
    parser.add_argument(
        "--network-type", 
        choices=["random", "small_world", "scale_free", "fully_connected"],
        default="fully_connected",
        help="Type of consumer network topology."
    )
    parser.add_argument(
        "--network-p", 
        type=float, 
        default=0.1, 
        help="Connection probability for random graph or rewiring probability for small-world."
    )
    parser.add_argument(
        "--network-k", 
        type=int, 
        default=4, 
        help="Number of neighbors for small-world or number of edges for new nodes in scale-free."
    )
    parser.add_argument(
        "--network-seed", 
        type=int, 
        default=None, 
        help="Random seed for network generation."
    )
    # --- End Network Arguments ---

    # 理智决策参数 - 分别控制每个决策步骤的rational状态
    parser.add_argument("--rational-share", action="store_true", help="Use rational decision for share decision")
    parser.add_argument("--rational-search", action="store_true", help="Use rational decision for search decision") 
    parser.add_argument("--rational-price", action="store_true", help="Use rational decision for price setting")

    parser.add_argument("--enable-cot", action="store_true", help="Enable Chain of Thought reasoning in LLM prompts")
    
    # 添加详细数据记录参数
    parser.add_argument("--record-detailed-data", action="store_true", 
                        help="Record detailed agent data to CSV files for information theory analysis")
    
    # 添加完整记忆日志参数
    parser.add_argument("--full-memory-log", action="store_true",
                        help="Log complete memory content for each agent (may generate very large log files)")
    parser.add_argument("--consumer-model-config", type=str, default="configs/consumer_model_config.json",
                        help="Path to per-consumer model config JSON (per-agent overrides)")
    parser.add_argument("--use-consumer-config", action="store_true",
                        help="Use per-consumer model config from JSON. If not set, all consumers use the same global model config.")
    return parser.parse_args()


def main(
        consumer_num: int = 4,
        firm_num: int = 6,
        search_cost: float = 0,
        agent_type: str = "llm",
        waiting_time: float = 3.0,
        use_dist: bool = False,
        num_rounds: int = 6,
        visualize: bool = False,
        threads: int = 4,
        memory_truncate: int = 3,
        memory_distill: bool = False,
        basic_price: float = 0.5,
        method: str = "adaptive",
        model_config_name: str = "gpt-config",
        model_names: Optional[dict] = None,
        force_fresh: bool = False,
        firm_cost: float = 0.0,
        top_k_consumers: int = 3,
        top_n_firms: int = 3,
        distill_broadcast: bool = False,
        start_broadcast_round: int = 2,
        broadcast_history_window: int = 1,
        network_type: str = "fully_connected",
        network_p: float = 0.1,
        network_k: int = 4,
        network_seed: Optional[int] = None,
        rational_share: bool = False,
        rational_search: bool = False,
        rational_price: bool = False,
        enable_cot: bool = False,
        record_detailed_data: bool = False,
        full_memory_log: bool = False,
        data_recorder_params: Optional[dict] = None,
        consumer_model_config_path:str = "configs/consumer_model_config.json",
        use_consumer_config: bool = False
) -> dict:

    try:
        if model_names is None:
            model_names = {
                'decide_share': model_config_name,
                'decide_search': model_config_name,
                'decide_purchase': model_config_name,
                'set_price': model_config_name,
                'set_personalized_price': model_config_name,
                'update_memory_distill': model_config_name,
                'distill_broadcast': model_config_name
            }

        print(f"DEBUG: 初始化模拟 - consumer_num={consumer_num}, firm_num={firm_num}, search_cost={search_cost}")
        print(f"DEBUG: 使用模型配置: {model_names}")
        print(f"DEBUG: Broadcast K={top_k_consumers}, N={top_n_firms}, Distill={distill_broadcast}")
        print(f"DEBUG: 广播配置 - 开始轮次={start_broadcast_round}, 历史窗口={broadcast_history_window}")
        print(f"DEBUG: Network Config - Type={network_type}, p={network_p}, k={network_k}, seed={network_seed}")
        print(f"DEBUG: Rational设置 - Share: {rational_share}, Search: {rational_search}, Price: {rational_price}")
        print(f"DEBUG: CoT Settings - Enable CoT: {enable_cot}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "configs/model_configs.json")
        print(f"DEBUG: attempting to load model configs from: {config_path}")

        agentscope.init(
            project="personalized_recommendation_simulation",
            name="main",
            save_code=False,
            save_api_invoke=True,
            model_configs=config_path,  
            use_monitor=False,
        )

        platform = Platform(
            search_cost=search_cost,
            memory_truncate=memory_truncate,
            memory_distill=memory_distill,
            model_config_name=model_config_name,
            model_names=model_names
        )

        # 设置rational决策所需的参数
        # v_dist参数需要与generate_valuations中的均匀分布区间一致
        # 在generate_valuations中，uniform分布默认使用np.random.uniform(0, 1)
        # 因此这里设置v_dist的low=0, high=1，使得CDF计算F_v = (v-low)/(high-low) = v
        v_dist = {'type': 'uniform', 'low': 0, 'high': 1}  # 估值分布参数，对应uniform分布
        r_value = 0.8  # 保留值，高于均值以体现消费者的谨慎性

        # 读配置文件
        consumer_models_map = {}
        try:
            cfg_path = consumer_model_config_path  # 由 args 传入或 main 的参数
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for entry in cfg.get("consumers", []):
                cid = entry.get("id")
                if cid is None:
                    continue
                try:
                    cid_int = int(cid)
                except Exception:
                    print(f"Warning: consumer id {cid} is not int; skipping.")
                    continue
                # models 字段应为 dict
                models = entry.get("models", {}) or {}
                if not isinstance(models, dict):
                    print(f"Warning: consumer {cid_int} models field not dict; skipping.")
                    continue
                consumer_models_map[cid_int] = models
            print(f"DEBUG: Loaded per-consumer model config for ids: {list(consumer_models_map.keys())}")
        except FileNotFoundError:
            print(f"DEBUG: consumer model config file not found: {consumer_model_config_path}. Using global defaults.")
        except Exception as e:
            print(
                f"DEBUG: failed to read consumer model config ({consumer_model_config_path}): {e}. Using global defaults.")
        #创建消费者
        if use_consumer_config:
            consumers = []
            for i in range(consumer_num):
                # 基于全局 model_names copy 一份作为 baseline（防止修改原 dict）
                merged = {}
                if model_names:
                    merged.update(model_names)
                # 合并 per-agent override（覆盖同名 key）
                per_models = consumer_models_map.get(i, {})
                if isinstance(per_models, dict):
                    merged.update(per_models)
                # merged 现在是传给 Consumer 的 model_names（可能为空 dict）
                c = Consumer(
                    index=i,
                    search_cost=platform.search_cost,
                    privacy_cost=round(random.uniform(0.025, 0.055), 3),
                    num_firms=firm_num,
                    dist_type='uniform',
                    memory_truncate=memory_truncate,
                    memory_distill=memory_distill,
                    model_config_name=model_config_name,
                    model_names=merged,
                    r_value=r_value,
                    v_dist=v_dist,
                    rational_search_cost=search_cost,
                    enable_cot=enable_cot
                )
                consumers.append(c)
        else:
            consumers = [
                Consumer(
                    index=i,
                    search_cost=platform.search_cost,
                    privacy_cost=round(random.uniform(0.025, 0.055), 3),
                    # 改为联合分布
                    num_firms=firm_num,
                    dist_type='uniform',
                    memory_truncate=memory_truncate,  # 只保留最近3轮的临时记忆
                    memory_distill=memory_distill,  # 启用记忆蒸馏
                    model_config_name=model_config_name,
                    model_names=model_names,
                    # rational决策所需参数
                    r_value=r_value,
                    v_dist=v_dist,
                    rational_search_cost=search_cost,
                    enable_cot=enable_cot
                    # 第一篇文章先不实现被拒绝后估计
                    # 改为联合分布 用条件期预测用户不选择分享数据的时候的数据
                )
                for i in range(consumer_num)
            ]
        firms = [
            Firm(
                index=i,
                method=method,
                memory_truncate=memory_truncate,
                memory_distill=memory_distill,
                basic_price=basic_price,
                pricing_mode='adaptive',
                firm_cost=firm_cost,
                model_config_name=model_config_name,
                model_names=model_names,
                # rational决策所需参数
                marginal_cost=firm_cost,  # 边际成本等于生产成本
                v_dist=v_dist,
                r_value=r_value,
                enable_cot=enable_cot
            )
            for i in range(firm_num)
        ]

        # Get consumers and firms
        platform.get_num_consumers(consumers)
        platform.get_num_firms(firms)

        # --- Generate Consumer Network Graph ---
        consumer_graph = None
        if network_type != "fully_connected":
            print(f"Generating consumer network: {network_type}")
            if network_type == "random":
                consumer_graph = nx.erdos_renyi_graph(consumer_num, network_p, seed=network_seed)
            elif network_type == "small_world":
                # Ensure k is even and less than n for watts_strogatz_graph
                k_adjusted = max(2, min(network_k, consumer_num - 1)) 
                if k_adjusted % 2 != 0:
                    k_adjusted = max(2, k_adjusted -1) # Make k even
                if consumer_num <= k_adjusted :
                    print(f"Warning: consumer_num <= k ({consumer_num}<={k_adjusted}), falling back to fully connected.")
                    consumer_graph = nx.complete_graph(consumer_num) # Fallback for safety
                else:
                     consumer_graph = nx.watts_strogatz_graph(consumer_num, k_adjusted, network_p, seed=network_seed)
            elif network_type == "scale_free":
                 # Ensure k (m in nx) is less than n
                k_adjusted = max(1, min(network_k, consumer_num-1))
                if consumer_num <= k_adjusted :
                    print(f"Warning: consumer_num <= k ({consumer_num}<={k_adjusted}), falling back to fully connected.")
                    consumer_graph = nx.complete_graph(consumer_num) # Fallback for safety
                else:
                    consumer_graph = nx.barabasi_albert_graph(consumer_num, k_adjusted, seed=network_seed)
            else:
                 raise ValueError(f"Unknown network type: {network_type}")
            # Ensure the graph is connected (optional, but often desirable)
            if not nx.is_connected(consumer_graph):
                 print("Warning: Generated graph is not connected. Using the largest connected component.")
                 largest_cc = max(nx.connected_components(consumer_graph), key=len)
                 consumer_graph = consumer_graph.subgraph(largest_cc).copy()
                 # Note: This might change the number of effective consumers if not handled carefully downstream.
                 # For simplicity now, we proceed, but a real implementation might need re-indexing or different handling.
        else:
            print("Using fully connected network model (global broadcast).")
        # --- End Network Generation ---

        # 初始化列表
        share_ratio_list = []
        consumer_surplus_list = []
        firm_surplus_list = []
        total_search_cost_list = []
        avg_search_cost_list = []  # 新增：平均搜索成本列表
        firm_prices_list = []  # 新增价格列表

        broadcaster = Broadcaster(
            top_k_consumers=top_k_consumers,
            top_n_firms=top_n_firms,
            distill_broadcast=distill_broadcast,
            start_broadcast_round=start_broadcast_round,
            broadcast_history_window=broadcast_history_window,
            model_config_name=model_config_name,
            model_names=model_names,
            consumer_graph=consumer_graph, # <-- Pass the graph
            enable_cot=enable_cot
        )

        # ===== 理性分享率均衡求解（仅在rational_share=True时执行）=====
        equilibrium_share_rate = None
        if rational_share:
            print("\n🎯 理性分享率均衡求解...")
            max_iter = 50
            tol = 1e-7
            σ = 0.4  # 初始分享率
            for iter_share in range(max_iter):
                # 所有消费者基于当前参数做分享决策
                share_decisions = []
                for consumer in consumers:
                    consumer.decide_share(rational=rational_share)
                    share_decisions.append(consumer.share)
                
                σ_new = np.mean(share_decisions)
                print(f"  迭代 {iter_share + 1}: σ = {σ:.4f} -> {σ_new:.4f}")
                
                if abs(σ_new - σ) < tol:
                    print(f"  分享率收敛于第 {iter_share + 1} 次迭代: σ = {σ_new:.4f}")
                    break
                σ = σ_new
            else:
                print(f"  注意：分享率未在 {max_iter} 次迭代内收敛，使用最终值: σ = {σ:.4f}")
            
            equilibrium_share_rate = σ
            print(f"🎯 理性均衡分享率: σ = {equilibrium_share_rate:.4f}")
            print("=" * 60)

        # 智能检查是否需要多线程：只有当所有三个关键步骤都是rational时才用单线程
        all_rational = rational_share and rational_search and rational_price
        effective_threads = 1 if all_rational else threads  # 只有完全rational才用单线程
        
        mode_desc = "fully rational (single thread)" if all_rational else f"mixed/LLM mode ({threads} threads)"
        print(f"DEBUG: Using {effective_threads} threads - {mode_desc}")
        print(f"DEBUG: Steps - Share: {'Rational' if rational_share else 'LLM'}, Search: {'Rational' if rational_search else 'LLM'}, Price: {'Rational' if rational_price else 'LLM'}")


        recorder = None
        if 'data_recorder_params' in locals() and data_recorder_params:
            recorder = DetailedDataRecorder(
                base_dir=data_recorder_params['base_dir'],
                exp_idx=data_recorder_params['exp_idx'],
                firm_num=firm_num,
                rational_tag=data_recorder_params['rational_tag']
            )

        with ThreadPoolExecutor(max_workers=effective_threads) as executor:
            # ===== 原有的回合模拟 =====
            for round_num in range(num_rounds):
                print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

                try:
                    should_broadcast = (round_num + 1) >= start_broadcast_round
                    broadcaster.generate_all_messages(num_consumers=consumer_num) 

                    if rational_share:
                        # 理性分享模式：使用预计算的均衡分享率，设置所有消费者的分享状态
                        print("📋 使用均衡分享决策...")
                        for consumer in consumers:
                            # 使用均衡分享率设置消费者分享状态
                            consumer.decide_share(rational=rational_share)
                    else:
                        # LLM分享模式：并行任务提交分享决策
                        consumer_tasks = []
                        for consumer in consumers:
                            broadcast_msg = ""
                            if should_broadcast:
                                 if broadcaster.network_type == "fully_connected":
                                    broadcast_msg = broadcaster.global_consumer_message
                                 else:
                                    broadcast_msg = broadcaster.messages_for_decide_share_per_consumer.get(consumer.index, "")
                            
                            consumer_tasks.append(executor.submit(
                                consumer.decide_share,
                                model_name=None,
                                broadcast_message=broadcast_msg,
                                rational=rational_share  # 使用分离的rational参数
                            ))
                        [task.result() for task in consumer_tasks]

                    share_ratio = sum(consumer.share for consumer in consumers) / consumer_num
                    print(f"Platform: Share ratio (σ) is now {share_ratio:.2f}")
                    
                    if rational_share:
                        print(f"理论均衡分享率: {equilibrium_share_rate:.4f}, 实际模拟分享率: {share_ratio:.4f}")
                    
                    list(executor.map(lambda f: f.get_num_consumers_and_firms(platform), firms))

                    # 检查是否第一轮，如果是则初始化企业的预期
                    for firm in firms:
                        firm.temp_memory['share_rate'] = round(share_ratio, 4)

                    # 更新企业预期：如果是理性分享，则使用均衡分享率更新所有企业的预期
                    if rational_share and equilibrium_share_rate is not None:
                        # 理性分享模式：所有企业使用相同的均衡分享率作为预期
                        for firm in firms:
                            firm.share_rate_predicted = equilibrium_share_rate
                            firm.temp_memory['share_rate_predicted'] = [equilibrium_share_rate, 'equilibrium']
                        print(f"所有企业预期已更新为均衡分享率: {equilibrium_share_rate:.4f}")
                    else:
                        # 非理性分享模式：企业根据历史数据更新预期
                        list(executor.map(lambda f: f.update_expectation(round_num), firms))

                    if rational_price:
                        # 理性价格模式：企业价格在均衡中确定
                        print("🎯 理性价格求解...")
                        # 价格均衡求解
                        max_iter = 50
                        tol = 1e-7
                        # 初始化价格（使用保留值附近的合理初值）
                        initial_prices = [max(0.1, r_value - 0.3) for _ in range(firm_num)]
                        current_prices = initial_prices.copy()
                        
                        for iter_price in range(max_iter):
                            market_price = np.mean(current_prices)
                            new_prices = []
                            
                            # 更新平台价格供企业定价时参考
                            platform.firm_prices = current_prices.copy()
                            
                            # 企业定价
                            for firm in firms:
                                # 传递当前分享率给企业
                                if rational_share and equilibrium_share_rate is not None:
                                    firm.temp_memory['actual_share_rate'] = equilibrium_share_rate
                                    firm.share_rate_predicted = equilibrium_share_rate
                                else:
                                    firm.temp_memory['actual_share_rate'] = share_ratio
                                    firm.share_rate_predicted = share_ratio

                                firm.num_firms = firm_num  # 确保企业知道总企业数
                                firm.set_price(rational=rational_price)
                                new_prices.append(firm.price)
                            
                            # 检查价格收敛
                            price_diff = np.max(np.abs(np.array(new_prices) - np.array(current_prices)))
                            print(f"  迭代 {iter_price + 1}: 价格差异 = {price_diff:.6f}, 均价 = {np.mean(new_prices):.4f}")
                            
                            if price_diff < tol:
                                print(f"  价格收敛于第 {iter_price + 1} 次迭代")
                                print(f"  最终价格: {[round(p, 4) for p in new_prices]}")
                                break
                                
                            current_prices = new_prices.copy()
                        else:
                            print(f"  价格未在 {max_iter} 次迭代内收敛")
                            print(f"  最终价格: {[round(p, 4) for p in current_prices]}")
                        
                        # 更新平台和企业的最终均衡状态
                        platform.firm_prices = current_prices.copy()
                        for i, firm in enumerate(firms):
                            firm.price = current_prices[i]
                            firm.temp_memory['equilibrium_price'] = current_prices[i]
                            if rational_share and equilibrium_share_rate is not None:
                                firm.temp_memory['equilibrium_share_rate'] = equilibrium_share_rate
                            else:
                                firm.temp_memory['equilibrium_share_rate'] = share_ratio
                    else:
                        # 非理性价格模式：并行处理企业定价 - 添加广播条件
                        list(
                        executor.map(
                            lambda f: f.set_price(
                                model_name=model_names.get('set_price'), 
                                broadcast_message=broadcaster.message_for_set_price if should_broadcast else "",
                                    rational=rational_price
                            ), 
                            firms
                        )
                    )
                        print("set price has finished")

                    platform.get_consumer_valuations(consumers)
                    platform.generate_search_sequence(consumers)
                    platform.get_firm_prices(firms)

                    if rational_search:
                        # 理性搜索模式：消费者搜索决策使用理性逻辑
                        print("使用理性搜索决策...")
                        for consumer in consumers:
                            consumer.decide_search(platform, rational=rational_search)
                    else:
                        # 非理性搜索模式：并行处理消费者搜索决策 - 添加广播条件
                        list(
                        executor.map(
                            lambda c: c.decide_search(
                                platform, 
                                model_name=model_names.get('decide_search'), 
                                broadcast_message="" if should_broadcast else "", # 取消search的广播
                                    rational=rational_search
                            ),
                            consumers
                        )
                    )

                    list(executor.map(lambda c: c.calculate_total_revenue_recommendation(), consumers))
                    list(executor.map(lambda c: c.update_memory(round_num, model_name=model_names.get('update_memory_distill')), consumers))

                    for consumer in consumers:
                        print(
                            f"\n--- Consumer {consumer.index + 1} ({'Shared Data' if consumer.share else 'Did Not Share Data'}) ---")
                        if consumer.share:
                            print(f"- Recommendation Order: {platform.search_sequence[consumer.index]}")

                        if consumer.purchase_index != -1:
                            goal_firms = [firm for firm in consumer.searched_firms if
                                          firm['index'] == consumer.purchase_index]
                            if goal_firms:  # 检查goal_firms不为空
                                print(f"Outcome: Purchased Product {consumer.purchase_index} for {goal_firms[0]['price']}.")
                            else:
                                # 如果goal_firms为空（就是rational决策导致的），从platform获取价格
                                if consumer.purchase_index < len(platform.firm_prices):
                                    price = platform.firm_prices[consumer.purchase_index]
                                    print(f"Outcome: Purchased Product {consumer.purchase_index} for {price}.")
                                else:
                                    print(f"Outcome: Purchased Product {consumer.purchase_index} (price unknown).")
                        else:
                            print("Outcome: Did not purchase any product.")

                    platform.get_consumer_purchase_behavior_recommendation(consumers)
                    platform.calculate_sales(firms)

                    for firm in firms:
                        firm_sales_details = platform.firm_sales.get(firm.index, {}).get('sales_details', {})
                        actual_share_ratio = equilibrium_share_rate if rational_share and equilibrium_share_rate is not None else share_ratio
                        firm.get_revenue(platform, actual_share_ratio, firm_sales_details)
                        firm.update_memory(round_num, model_name=model_names.get('update_memory_distill'))

                    print("[DEBUG] Temp memories BEFORE broadcaster collection:")
                    for c in consumers:
                        print(f"  Consumer {c.index} temp_mem: {c.temp_memory}")
                        print(f"Consumer {c.index}: {json.dumps(c.model_names, ensure_ascii=False)}")
                    for f in firms: print(f"  Firm {f.index} temp_mem: {f.temp_memory}")

                    #在回合结束时，让 broadcaster 收集所有 agent 的 temp_memory
                    broadcaster.get_last_round_mems(consumers, firms)

                    platform.calculate_surplus(consumers, firms)
                    
                    # 试试直接使用Consumer对象中已经正确计算的总搜索成本？
                    total_search_cost = sum(c.total_search_cost for c in consumers)
                    # 计算平均搜索成本：总搜索成本除以消费者数量
                    avg_search_cost = total_search_cost / consumer_num if consumer_num > 0 else 0
                    print(f"Total Search Cost: {total_search_cost}")
                    print(f"Average Search Cost: {avg_search_cost}")  
                    print(f"consumer_num: {consumer_num}")

                    # 更新列表
                    share_ratio_list.append(share_ratio)
                    consumer_surplus_list.append(platform.consumer_surplus)
                    firm_surplus_list.append(platform.firm_surplus)
                    total_search_cost_list.append(total_search_cost)
                    avg_search_cost_list.append(avg_search_cost)  
                    firm_prices_list.append(platform.firm_prices)  

                    print(f"\nDebug - Round {round_num + 1} values:")
                    print(f"Share Ratio: {share_ratio}")
                    print(f"Consumer Surplus: {platform.consumer_surplus}")
                    print(f"Firm Surplus: {platform.firm_surplus}")
                    print(f"Total Search Cost: {total_search_cost}")
                    print(f"Average Search Cost: {avg_search_cost}")  

                    platform.update_memory(round_num, model_name=model_names.get('update_memory_distill'))

                    list(executor.map(lambda f: f.reset(), consumers))
                    list(executor.map(lambda f: f.reset(), firms))

                    print(f"\n--- Round {round_num + 1} Results ---")
                    print(f"Consumer Surplus: {platform.consumer_surplus}")
                    print(f"Platform: Share ratio (σ) is now {share_ratio:.2f}")
                    print(f"Prices: {platform.firm_prices}")
                    print(f"*Firm Surplus: {platform.firm_surplus}")

                    # 如果启用了完整记忆日志，打印所有agent的记忆
                    if data_recorder_params and data_recorder_params.get('full_memory_log', False):
                        print(f"\n--- Full Memory Log for Round {round_num + 1} ---")
                        print("=== Consumer Memories ===")
                        for consumer in consumers:
                            print(f"\nConsumer {consumer.index}:")
                            print(f"  History Memory: {consumer.history_memory}")
                            print(f"  Temp Memory: {consumer.temp_memory}")
                        
                        print("\n=== Firm Memories ===")
                        for firm in firms:
                            print(f"\nFirm {firm.index}:")
                            print(f"  History Memory: {firm.history_memory}")
                            print(f"  Temp Memory: {firm.temp_memory}")
                        
                        print("\n=== Platform Memory ===")
                        print(f"  History Memory: {platform.history_memory}")
                        print(f"  Temp Memory: {platform.temp_memory}")
                        print("--- End of Full Memory Log ---\n")

                    platform.reset()

                    # 在每轮结束时记录数据
                    if recorder:
                        recorder.record_round_data(round_num, consumers, firms, platform, share_ratio)

                except Exception as e:
                    print(f"\n回合 {round_num + 1} 出现错误: {str(e)}")
                    import traceback
                    traceback.print_exc()

        if rational_share and equilibrium_share_rate is not None:
            print(f"\n🎉 理性分享模拟完成!")
            print(f"均衡分享率: {equilibrium_share_rate:.4f}")
            print(f"平均价格: {np.mean(platform.firm_prices):.4f}")

        print(f"\n--- Simulation Results (End of Experiment) ---")  
        print(f"Share Ratio: {share_ratio_list}")
        print(f"Firm Prices: {firm_prices_list}")
        print(f"Consumer Surplus: {consumer_surplus_list}")
        print(f"Firm Surplus: {firm_surplus_list}")
        print(f"Total Search Cost: {total_search_cost_list}")
        print(f"Average Search Cost: {avg_search_cost_list}")  

        return {
            'share_ratio': share_ratio_list,
            'consumer_surplus': consumer_surplus_list,
            'firm_surplus': firm_surplus_list,
            'total_search_cost': total_search_cost_list,
            'avg_search_cost': avg_search_cost_list,  
            'firm_prices': firm_prices_list  
        }
    except Exception as e:
        print(f"\n模拟过程中出现严重错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'share_ratio': share_ratio_list if 'share_ratio_list' in locals() else [],
            'consumer_surplus': consumer_surplus_list if 'consumer_surplus_list' in locals() else [],
            'firm_surplus': firm_surplus_list if 'firm_surplus_list' in locals() else [],
            'total_search_cost': total_search_cost_list if 'total_search_cost_list' in locals() else [],
            'avg_search_cost': avg_search_cost_list if 'avg_search_cost_list' in locals() else [],  # 新增：返回平均搜索成本
            'firm_prices': firm_prices_list if 'firm_prices_list' in locals() else []  # 新增返回价格数据
        }


def run_multiple_experiments(num_experiments=6, clean_start=False, start_firm_num=1, force_fresh=False, **kwargs):
    """运行多次实验并保存可视化结果，每次实验使用不同的firm_num"""
    
    matplotlib.use('Agg', force=True)
    plt.ioff() 

    plt.style.use('default')  # 使用默认样式作为基础
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'DejaVu Sans',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': 300
    })
    
    def generate_rational_title_suffix(**params):
        """生成基于rational参数的标题后缀"""
        rational_share = params.get('rational_share', False)
        rational_search = params.get('rational_search', False) 
        rational_price = params.get('rational_price', False)
        
        has_any_rational = rational_share or rational_search or rational_price
        
        if not has_any_rational:
            return " - LLM Simulation"
        
        # 构建rational标签
        rational_parts = []
        if rational_share:
            rational_parts.append("Share")
        if rational_search:
            rational_parts.append("Search")
        if rational_price:
            rational_parts.append("Price")
        
        if rational_parts:
            return f" - Rational: {', '.join(rational_parts)}"
        else:
            return " - LLM Simulation"
    
    original_stdout_f = sys.stdout  
    log_file_path = None
    log_file_handle = None

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pid = os.getpid()
        base_results_dir = 'recommendation_experiment_results'
        
        rational_tag = "rational" if any([kwargs.get('rational_share', False),
                                        kwargs.get('rational_search', False),
                                        kwargs.get('rational_price', False)]) else "llm"
        
        run_dir_name = f"run_{timestamp}_pid{pid}_{rational_tag}_startfirm{start_firm_num}_numexp{num_experiments}"
        run_results_dir = os.path.join(base_results_dir, run_dir_name)
        checkpoint_dir = os.path.join(run_results_dir, 'checkpoints')

        print(f"DEBUG: Base results directory: {base_results_dir}")
        print(f"DEBUG: Process ID (PID): {pid}")
        print(f"DEBUG: Rational mode: {rational_tag}")
        print(f"DEBUG: Results for this run will be saved in: {run_results_dir}")
        print(f"DEBUG: Checkpoints for this run will be saved in: {checkpoint_dir}")

        if clean_start and os.path.exists(base_results_dir):
            print(f"--clean_start specified. Deleting base results directory: {base_results_dir}")
            try:
                shutil.rmtree(base_results_dir)
                print(f"Successfully deleted directory: {base_results_dir}")
            except Exception as e:
                print(f"Error deleting base directory: {str(e)}")

        if not os.path.exists(run_results_dir):
            os.makedirs(run_results_dir)
            print(f"DEBUG: Created run results directory: {run_results_dir}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"DEBUG: Created checkpoints directory: {checkpoint_dir}")

        log_file_path = os.path.join(run_results_dir, f"run_{timestamp}_pid{pid}_{rational_tag}.log")
        print(f"DEBUG: Print output will be logged to: {log_file_path}",
              file=original_stdout_f)  

        log_file_handle = open(log_file_path, 'w', encoding='utf-8')
        sys.stdout = log_file_handle

        print(f"Starting run_multiple_experiments with following base config:")
        print(f"num_experiments={num_experiments}, start_firm_num={start_firm_num}")
        print(f"clean_start={clean_start}, force_fresh={force_fresh}")
        print(f"Base kwargs: {kwargs}")
        print(f"Results Dir: {run_results_dir}")
        print("---")

        all_results = []
        checkpoint_file = os.path.join(checkpoint_dir, 'experiment_checkpoint.pkl')
        experiment_offset = 0  

        if os.path.exists(checkpoint_file) and not clean_start and not force_fresh:
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    all_results = checkpoint_data['results']
                    experiment_offset = checkpoint_data.get('next_exp_offset', 0)
                    print(f"Resuming from checkpoint. Already completed {experiment_offset} experiments in this run.")
                    print(f"DEBUG: Loaded {len(all_results)} results from previous runs/checkpoints.")
            except Exception as e:
                print(f"Error reading checkpoint file '{checkpoint_file}': {str(e)}. Starting fresh for this run.")
                all_results = []
                experiment_offset = 0
        else:
            if force_fresh:
                print("--force-fresh specified. Ignoring any existing checkpoint and starting fresh for this run.")
            elif not clean_start:
                print(
                    f"No checkpoint found at '{checkpoint_file}' or --force-fresh specified. Starting fresh for this run.")
            all_results = []
            experiment_offset = 0

        metrics = ['share_ratio', 'consumer_surplus', 'firm_surplus', 'avg_search_cost', 'avg_price']
        titles = ['Share Ratio', 'Consumer Surplus', 'Firm Surplus', 'Average Search Cost', 'Average Price']

        # 创建跨实验数据的CSV文件，用于存储所有实验的平均值
        inter_exp_csv_path = os.path.join(run_results_dir, f'inter_experiment_avg_data_{rational_tag}.csv')
        with open(inter_exp_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            rational_info = generate_rational_title_suffix(**kwargs).strip(' -')
            writer.writerow(['# Experiment Configuration:'] + [rational_info])
            writer.writerow(['Experiment', 'Firm_Num'] + metrics)

        def save_checkpoint(current_offset, results):
            try:
                checkpoint_data = {
                    'results': results,
                    'next_exp_offset': current_offset + 1  
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)

                json_friendly_results = []
                for result in results:
                    json_result = {}
                    for k, v in result.items():
                        if isinstance(v, np.ndarray):
                            json_result[k] = v.tolist()
                        elif isinstance(v, list) and all(isinstance(item, (int, float, np.number)) for item in v):
                            json_result[k] = [float(item) for item in v]
                        else:
                            json_result[k] = v
                    json_friendly_results.append(json_result)

                json_results_path = os.path.join(run_results_dir, 'experiment_results.json')
                with open(json_results_path, 'w') as f:
                    json.dump(json_friendly_results, f, indent=2)

                print(f"Checkpoint saved for experiment offset {current_offset} in '{checkpoint_dir}'")
                print(f"JSON results updated at '{json_results_path}'")
            except Exception as e:
                print(f"Error saving checkpoint or JSON results: {str(e)}")
                import traceback
                traceback.print_exc()

        for exp_idx in range(experiment_offset, num_experiments):
            current_firm_num = start_firm_num + exp_idx
            current_run_params = kwargs.copy()
            current_run_params['firm_num'] = current_firm_num
            current_run_params['firm_cost'] = kwargs.get('firm_cost', 0.0)
            current_run_params['force_fresh'] = force_fresh
            current_run_params['top_k_consumers'] = kwargs.get('top_k_consumers', 3)
            current_run_params['top_n_firms'] = kwargs.get('top_n_firms', 3)
            current_run_params['distill_broadcast'] = kwargs.get('distill_broadcast', False)
            current_run_params['start_broadcast_round'] = kwargs.get('start_broadcast_round', 2) 
            current_run_params['broadcast_history_window'] = kwargs.get('broadcast_history_window', 1) 
            current_run_params['rational_share'] = kwargs.get('rational_share', False)
            current_run_params['rational_search'] = kwargs.get('rational_search', False) 
            current_run_params['rational_price'] = kwargs.get('rational_price', False)
            current_run_params['enable_cot'] = kwargs.get('enable_cot', False)
            current_run_params['record_detailed_data'] = kwargs.get('record_detailed_data', False)
            current_run_params['full_memory_log'] = kwargs.get('full_memory_log', False)
            
            # 只有在启用详细数据记录时才添加数据记录器参数
            if kwargs.get('record_detailed_data', False):
                current_run_params['data_recorder_params'] = {
                    'base_dir': run_results_dir,
                    'exp_idx': exp_idx,
                    'rational_tag': rational_tag,
                    'full_memory_log': kwargs.get('full_memory_log', False)  # 添加完整记忆日志参数
                }

            print(f"\nRunning Experiment {exp_idx + 1}/{num_experiments} (firm_num={current_firm_num})")
            print(f"Passing params to main: {current_run_params}")

            try:
                result = main(**current_run_params)

                if not isinstance(result, dict):
                    print(f"Warning: Experiment {exp_idx + 1} returned invalid result type: {type(result)}")
                    continue

                avg_prices_this_exp = []
                if 'firm_prices' in result and result['firm_prices']:
                    for round_prices in result['firm_prices']:
                        if round_prices:  
                            avg_prices_this_exp.append(round(sum(round_prices) / len(round_prices), 4))
                        else:
                            avg_prices_this_exp.append(0.0)  
                result['avg_price'] = avg_prices_this_exp

                for metric in metrics:
                    if metric in result and isinstance(result[metric], list):
                        try:
                            if metric == 'avg_price':  # 价格相关指标保留4位小数
                                result[metric] = [round(float(x), 4) if isinstance(x, (int, float, np.number)) else x for x in result[metric]]
                            else:
                                result[metric] = [round(float(x), 4) if isinstance(x, (int, float, np.number)) else x for x in result[metric]]
                        except (TypeError, ValueError) as round_err:
                            print(
                                f"Warning: Could not round metric '{metric}' for exp {exp_idx + 1}. Data: {result[metric]}. Error: {round_err}")

                result['_exp_params'] = {'experiment_index': exp_idx + 1, 'firm_num': current_firm_num}
                all_results.append(result)

                save_checkpoint(exp_idx, all_results)

                print(f"Experiment {exp_idx + 1} Results Summary:")
                for metric in metrics:
                    if metric in result and result[metric]:  # Check if list is not empty
                        try:
                            if metric == 'avg_price':  # 价格相关指标显示4位小数
                                avg_value = round(np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                                print(f"  Avg {metric}: {avg_value:.4f}")
                            else:
                                avg_value = round(
                                    np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                            print(f"  Avg {metric}: {avg_value}")
                        except Exception as mean_err:
                            print(
                                f"  Warning: Could not calculate mean for {metric}. Data: {result[metric]}. Error: {mean_err}")
                    else:
                        print(f"  Metric {metric}: No data or empty list.")


                intra_exp_csv_path = os.path.join(run_results_dir, f'experiment_{exp_idx + 1}_firms_{current_firm_num}_{rational_tag}_data.csv')
                with open(intra_exp_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    rational_info = generate_rational_title_suffix(**current_run_params).strip(' -')
                    writer.writerow(['# Experiment Configuration:'] + [rational_info])
                    writer.writerow(['# Firm Count:'] + [current_firm_num])
                    writer.writerow(['Round'] + metrics)  
                    
                    max_rounds = 0
                    for metric in metrics:
                        if metric in result and result[metric]:
                            max_rounds = max(max_rounds, len(result[metric]))
                    
                    for round_idx in range(max_rounds):
                        row = [round_idx + 1]  
                        for metric in metrics:
                            if metric in result and result[metric] and round_idx < len(result[metric]):
                                row.append(result[metric][round_idx])
                            else:
                                row.append('')  
                        writer.writerow(row)
                
                print(f"Experiment {exp_idx + 1} data saved to CSV: {intra_exp_csv_path}")


                with open(inter_exp_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [exp_idx + 1, current_firm_num]
                    
                    for metric in metrics:
                        if metric in result and result[metric]:
                            try:
                                if metric == 'avg_price':  # 价格相关指标保留4位小数
                                    avg_value = round(np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                                    row.append(avg_value)
                                else:
                                    avg_value = round(
                                        np.mean([x for x in result[metric] if isinstance(x, (int, float, np.number))]), 4)
                                    row.append(avg_value)
                            except Exception:
                                row.append('')  
                        else:
                            row.append('')  
                    
                    writer.writerow(row)

                if kwargs.get('visualize', False):
                    title_suffix = generate_rational_title_suffix(**current_run_params)
                    
                    fig_intra, axs_intra = plt.subplots(2, 3, figsize=(18, 12))
                    fig_intra.suptitle(f'Experiment {exp_idx + 1} Results (Firm Num: {current_firm_num}){title_suffix}', 
                                     fontsize=16, fontweight='bold')
                    axs_intra = axs_intra.flatten()

                    plot_metrics_intra = metrics  
                    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC']  
                    
                    for j, metric in enumerate(plot_metrics_intra):
                        ax = axs_intra[j]
                        if metric in result and result[metric]:
                            numeric_data = [x for x in result[metric] if isinstance(x, (int, float, np.number))]
                            if numeric_data:
                                ax.plot(range(1, len(numeric_data) + 1), numeric_data, 
                                       color=colors[j], linewidth=2.5, marker='o', markersize=6, 
                                       markerfacecolor='white', markeredgewidth=2, alpha=0.8)
                            else:
                                ax.text(0.5, 0.5, 'No numeric data', horizontalalignment='center',
                                        verticalalignment='center', transform=ax.transAxes, 
                                        fontsize=12, color='red')
                        else:
                            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center',
                                    transform=ax.transAxes, fontsize=12, color='red')
                        
                        ax.set_title(titles[j], fontsize=14, fontweight='bold', pad=15)
                        ax.set_xlabel('Round', fontsize=12)
                        ax.set_ylabel('Value', fontsize=12)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(1.5)
                        ax.spines['bottom'].set_linewidth(1.5)


                    if len(plot_metrics_intra) == 5:
                        axs_intra[-1].set_visible(False)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
                    

                    intra_plot_path = os.path.join(run_results_dir,
                                                   f'experiment_{exp_idx + 1}_firms_{current_firm_num}_{rational_tag}.png')
                    plt.savefig(intra_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_intra)
                    print(f"Intra-experiment plot saved to: {intra_plot_path}")

                if kwargs.get('visualize', False):
                    title_suffix = generate_rational_title_suffix(**current_run_params)
                    
                    fig_inter, axs_inter = plt.subplots(2, 3, figsize=(18, 12))
                    fig_inter.suptitle(f'Average Results Across Experiments (Up to Exp {exp_idx + 1}){title_suffix}', 
                                      fontsize=16, fontweight='bold')
                    axs_inter = axs_inter.flatten()

                    plot_metrics_inter = metrics
                    firm_nums_completed = [r['_exp_params']['firm_num'] for r in all_results if '_exp_params' in r]
                    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC']  

                    for j, metric in enumerate(plot_metrics_inter):
                        ax = axs_inter[j]
                        avg_values = []
                        valid_firm_nums = []
                        for k, res in enumerate(all_results):
                            if metric in res and res[metric]:
                                try:
                                    numeric_metric_data = [x for x in res[metric] if
                                                           isinstance(x, (int, float, np.number))]
                                    if numeric_metric_data:
                                        avg_values.append(round(np.mean(numeric_metric_data), 4))
                                        if '_exp_params' in res:
                                            valid_firm_nums.append(res['_exp_params']['firm_num'])
                                except Exception as avg_err:
                                    print(
                                        f"Warning: Could not calculate average for {metric} in experiment {k + 1}. Error: {avg_err}")

                        if avg_values and valid_firm_nums:
                            ax.plot(valid_firm_nums, avg_values, color=colors[j], linewidth=3, 
                                   marker='o', markersize=8, markerfacecolor='white', 
                                   markeredgewidth=2.5, alpha=0.9, label=titles[j])
                            
                            # for x, y in zip(valid_firm_nums, avg_values):
                            #     if j == 4:  # avg_price是第5个指标（索引4）
                            #         ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                            #                   xytext=(0,10), ha='center', fontsize=9, alpha=0.8)
                            #     else:
                            #         ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                            #                   xytext=(0,10), ha='center', fontsize=9, alpha=0.8)
                        else:
                            ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center',
                                    verticalalignment='center', transform=ax.transAxes,
                                    fontsize=12, color='red')
                        
                        ax.set_title(titles[j], fontsize=14, fontweight='bold', pad=15)
                        ax.set_xlabel('Number of Firms', fontsize=12)
                        ax.set_ylabel('Average Value', fontsize=12)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(1.5)
                        ax.spines['bottom'].set_linewidth(1.5)
                        
                        # 设置x轴刻度为整数
                        if valid_firm_nums:
                            ax.set_xticks(range(min(valid_firm_nums), max(valid_firm_nums) + 1))

                    # Hide the last empty subplot if metrics count is 5
                    if len(plot_metrics_inter) == 5:
                        axs_inter[-1].set_visible(False)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    # 改进文件命名
                    inter_plot_path = os.path.join(run_results_dir, f'average_results_up_to_exp_{exp_idx + 1}_{rational_tag}.png')
                    plt.savefig(inter_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_inter)
                    print(f"Inter-experiment average plot saved to: {inter_plot_path}")

            except Exception as e:
                print(f"Experiment {exp_idx + 1} failed: {str(e)}")
                print("\nDetailed error trace:")
                import traceback
                traceback.print_exc()

                error_file = os.path.join(checkpoint_dir, f'error_experiment_{exp_idx + 1}_{rational_tag}.txt')
                with open(error_file, 'w') as f:
                    f.write(f"Experiment {exp_idx + 1} (firm_num={current_firm_num}) Error: {str(e)}\n")
                    f.write(f"Rational Mode: {generate_rational_title_suffix(**current_run_params).strip(' -')}\n\n")
                    traceback.print_exc(file=f)

                # Save checkpoint even on error to preserve progress
                save_checkpoint(exp_idx, all_results)

        # Final calculation of average results (optional, as plots are generated incrementally)
        avg_results = {}
        if not all_results:
            print("No experiments completed successfully.")
            return avg_results

        for metric in metrics:
            try:
                metric_data_all_exp = []
                for r in all_results:
                    if metric in r and isinstance(r[metric], list):
                        # Collect *means* from each experiment's rounds
                        numeric_data = [x for x in r[metric] if isinstance(x, (int, float, np.number))]
                        if numeric_data:
                            metric_data_all_exp.append(np.mean(numeric_data))
                        else:
                            metric_data_all_exp.append(np.nan)  # Use NaN for experiments with no numeric data

                if not any(np.isnan(x) for x in metric_data_all_exp):
                    if metric == 'avg_price':  # 价格相关指标保留4位小数
                        avg_results[metric] = [round(float(x), 4) for x in metric_data_all_exp]
                        overall_avg = round(np.nanmean(metric_data_all_exp), 4)
                        print(f"Overall Average for {metric}: {overall_avg:.4f}")
                        avg_results[f"{metric}_overall_avg"] = overall_avg
                    else:
                        avg_results[metric] = [round(float(x), 4) for x in metric_data_all_exp]
                        overall_avg = round(np.nanmean(metric_data_all_exp), 4)
                    print(f"Overall Average for {metric}: {overall_avg}")
                    avg_results[f"{metric}_overall_avg"] = overall_avg
                else:
                    print(f"Could not compute overall average for {metric} due to missing data.")
                    avg_results[metric] = []  # Or handle as needed
            except Exception as e:
                print(f"Error calculating final average for {metric}: {str(e)}")
                avg_results[metric] = []

        print(f"\nAll {num_experiments} experiments finished. Results saved in: {run_results_dir}")
        
        # 创建实验总结文件
        summary_path = os.path.join(run_results_dir, f'experiment_summary_{rational_tag}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # 配置信息
            f.write("CONFIGURATION:\n")
            f.write(f"- Decision Mode: {generate_rational_title_suffix(**kwargs).strip(' -')}\n")
            f.write(f"- Number of Experiments: {num_experiments}\n")
            f.write(f"- Firm Range: {start_firm_num} to {start_firm_num + num_experiments - 1}\n")
            f.write(f"- Consumer Count: {kwargs.get('consumer_num', 'N/A')}\n")
            f.write(f"- Rounds per Experiment: {kwargs.get('num_rounds', 'N/A')}\n")
            f.write(f"- Search Cost: {kwargs.get('search_cost', 'N/A')}\n")
            f.write(f"- Network Type: {kwargs.get('network_type', 'N/A')}\n\n")
            
            # 结果概要
            f.write("RESULTS OVERVIEW:\n")
            for metric in metrics:
                if metric in avg_results and f"{metric}_overall_avg" in avg_results:
                    if metric == 'avg_price':  # 价格相关指标显示4位小数
                        f.write(f"- Average {metric.replace('_', ' ').title()}: {avg_results[f'{metric}_overall_avg']:.4f}\n")
                    else:
                        f.write(f"- Average {metric.replace('_', ' ').title()}: {avg_results[f'{metric}_overall_avg']}\n")
            
            f.write(f"\nFiles Generated:\n")
            f.write(f"- Raw Data: inter_experiment_avg_data_{rational_tag}.csv\n")
            f.write(f"- Visualizations: *_{rational_tag}.png\n")
            f.write(f"- Individual Data: experiment_*_{rational_tag}_data.csv\n")
            f.write(f"- Log File: run_*_{rational_tag}.log\n")
            
            f.write(f"\nTimestamp: {timestamp}\n")
            f.write(f"Process ID: {pid}\n")
        
        print(f"Experiment summary saved to: {summary_path}")
        return avg_results
    except Exception as e:
        print(f"\nFATAL ERROR in run_multiple_experiments: {str(e)}", file=original_stdout_f)
        import traceback
        traceback.print_exc(file=original_stdout_f)
        # If logging started, also try logging the error
        if log_file_handle and not log_file_handle.closed:
            print(f"\nFATAL ERROR in run_multiple_experiments: {str(e)}", file=log_file_handle)
            traceback.print_exc(file=log_file_handle)

    finally:
        log_file_closed_properly = False
        # Restore stdout and close log file
        if log_file_handle and not log_file_handle.closed:  # Check if handle exists and is open
            # Check if stdout was actually redirected before restoring
            if sys.stdout == log_file_handle:
                sys.stdout = original_stdout_f
                print(f"Finished logging. Output redirected back to console.", file=original_stdout_f)
            else:
                print(f"Log file handle existed but stdout wasn\'t redirected to it upon finally. Closing file.",
                      file=original_stdout_f)
            log_file_handle.close()
            log_file_closed_properly = True # Mark as closed
        elif log_file_path:
            # If handle creation failed but path exists, mention it
            print(f"Log file handle was not properly created or opened for {log_file_path}", file=original_stdout_f)
        else:
            print(f"Logging was not initiated.", file=original_stdout_f)

        # 清理matplotlib资源，避免tkinter错误
        try:
            plt.close('all')  # 关闭所有图形
            matplotlib.pyplot.clf()  # 清理当前图形
            matplotlib.pyplot.cla()  # 清理当前轴
        except Exception as e:
            print(f"matplotlib清理时出现错误: {e}", file=original_stdout_f)

        # 清理日志文件，删除所有包含"INFO"或"WARNING"的行
        if log_file_closed_properly and log_file_path and os.path.exists(log_file_path):
            print(f"清理日志文件: {log_file_path}", file=original_stdout_f)
            try:
                # 读取原始日志
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 过滤掉包含INFO或WARNING的行
                filtered_lines = []
                for line in lines:
                    if 'INFO' not in line and 'WARNING' not in line:
                        filtered_lines.append(line)
                
                # 将过滤后的内容写回原文件
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(filtered_lines)
                
                # 同时保存一份原始日志
                original_log_path = log_file_path + '.original'
                with open(original_log_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print(f"日志清理完成。原始日志备份至: {original_log_path}", file=original_stdout_f)
                print(f"过滤前行数: {len(lines)}, 过滤后行数: {len(filtered_lines)}", file=original_stdout_f)
            except Exception as e:
                print(f"日志清理失败: {str(e)}", file=original_stdout_f)


if __name__ == "__main__":
    args = parse_args()

    # 设置不同步骤使用的模型
    model_names = {
        'decide_share': 'gpt-config',
        'decide_search': 'gpt-config',
        'decide_purchase': 'gpt-config',
        'set_price': 'gpt-config',
        'set_personalized_price': 'gpt-config', # 情况2，目前不予考虑
        'update_memory_distill': 'gpt-config',# 蒸馏记忆
        'distill_broadcast': 'gpt-config'# 蒸馏广播内容
    }

    # 检查是否有命令行参数指定了备用模型
    fallback_model = args.model_config_name

    print(f"\n### 使用模型配置: {model_names} (备用模型: {fallback_model}) ###\n")

    if args.clean_start:
        print("将删除之前的所有实验记录，从头开始实验...")

    print("开启详细调试模式，显示所有错误和警告...")

    try:
        experiment_params = {
            "consumer_num": args.consumer_num,
            "start_firm_num": args.firm_num,
            "search_cost": args.search_cost,
            "agent_type": args.agent_type,
            "use_dist": args.use_dist,
            "num_rounds": args.num_rounds,
            "visualize": args.visualize,
            "threads": args.threads,
            "memory_truncate": args.memory_truncate,
            "memory_distill": args.memory_distill,
            "model_config_name": fallback_model,
            "model_names": model_names,
            "force_fresh": args.force_fresh,
            "firm_cost": args.firm_cost,
            "method": args.pricing_mode,
            "top_k_consumers": args.top_k_consumers,
            "top_n_firms": args.top_n_firms,
            "distill_broadcast": args.distill_broadcast,
            "start_broadcast_round": args.start_broadcast_round,  # 新增参数
            "broadcast_history_window": args.broadcast_history_window,  # 新增参数
            "network_type": args.network_type,
            "network_p": args.network_p,
            "network_k": args.network_k,
            "network_seed": args.network_seed,
            "rational_share": args.rational_share,
            "rational_search": args.rational_search,
            "rational_price": args.rational_price,
            "enable_cot": args.enable_cot,
            "record_detailed_data": args.record_detailed_data,  # 详细数据记录参数
            "full_memory_log": args.full_memory_log,  # 完整记忆日志参数
            "consumer_model_config_path": args.consumer_model_config,
            "use_consumer_config": args.use_consumer_config,
        }

        run_multiple_experiments(
            num_experiments=args.num_experiments,
            clean_start=args.clean_start,
            **experiment_params
        )
    except KeyboardInterrupt:
        print("\n程序被用户中断。已保存检查点，可以稍后恢复。")
    except Exception as e:
        print(f"\n程序遇到异常: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n程序结束。")
        plt.close('all')
