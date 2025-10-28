import random
import re
import time
from random import choice
from typing import Optional, Sequence, Union, List, Dict, Any
import warnings
import logging
import os
import copy
import networkx as nx
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

# 禁用所有警告
warnings.filterwarnings("ignore")

# 禁用 INFO 和 WARNING 级别的日志
logging.getLogger().setLevel(logging.ERROR)

# 禁用 agentscope 的日志
os.environ["AGENTSCOPE_LOG_LEVEL"] = "ERROR"

from loguru import logger

# 配置 loguru 日志级别
logger.remove() # 移除默认的处理器
logger.add(lambda msg: None, level="ERROR") # 只显示 ERROR 级别以上的日志

from agentscope.agents import AgentBase
from agentscope.message import Msg

# 定义消费者类型，属性有隐私成本、公司数量、对公司产品的估值向量（维数等于公司数量）
class Consumer(AgentBase):
    def __init__(self, search_cost, index, privacy_cost, num_firms, dist_type, dist_params=None,
                 memory_truncate=-1, memory_distill=False, model_config_name="gpt-config", model_names=None,
                 r_value=None, v_dist=None, rational_search_cost=None, enable_cot=False):
        self.model_names= model_names or {}
        self.history_memory = {} 
        self.model_config_name = model_config_name
        super().__init__(name=f'consumer_{index}', model_config_name=model_config_name, use_memory=True)
        self.index = index
        self.search_cost = search_cost
        self.privacy_cost = privacy_cost
        self.num_firms = num_firms
        self.share = False
        self.valuations = self.generate_valuations(num_firms,dist_type, dist_params)
        self.quality_consciousness = self.generate_quality_consciousness(dist_type, dist_params)
        # 当期购买的产品对应的公司序号（-1表示不购买）
        self.purchase_index = -1
        self.total_search_cost = 0
        self.total_revenue = 0
        # 消费者已经搜索过的公司及其对应的估值和价格
        self.searched_firms = []  # 解释该嵌套结构：每个元素是一个字典，包含了消费者搜索过的一家公司的信息，包括序号、估值、价格
        # 记忆模块，存储之前消费者的所有行为，包括是否分享数据、是否收到推荐、搜索过的公司、价格、估值、购买行为、总收益及这些行为对应的理由，请选择合适的嵌套结构存储这些信息
        self.history_memory = {}  # 解释该嵌套结构：每个值是一个字典，包含了消费者的一次行为，包括是否分享数据、是否收到推荐、搜索过的公司、价格、估值、购买行为及这些行为对应的理由
        # 每一个行为都要和理由对应，所以字典的键值对应关系是：环节-【行为，理由】
        # 消费者的当期临时记忆，格式同memory的元素
        self.temp_memory = {}  # 确保初始化为空字典
        self.memory_distill_text = []  # 存储蒸馏后的记忆
        self.memory_truncate = memory_truncate
        self.memory_distill = memory_distill
        self.model_names = model_names
        self.enable_cot = enable_cot
        self.utility=0#情形三消费者收益

        # 理性决策逻辑参数，提供默认值
        self.r = r_value if r_value is not None else 0.8  # 保留值，默认0.8
        self.v_dist = v_dist if v_dist is not None else {'low': 0, 'high': 1}  # 估值分布参数，默认uniform[0,1]
        self.s = rational_search_cost if rational_search_cost is not None else search_cost  # 理性搜索成本
        self.idx = index  # 索引属性
        self.τ = privacy_cost  # 隐私成本属性
        self.n = num_firms  # 企业数量属性
        self.utility = 0.0  # 储消费者最终效用
        self.purchase_from = None  #记录购买的企业编号
        self.search_times = 0  # 记录搜索次数
        self.prompt = Msg(
            name="user",
            role="user",
            content="You are a consumer. You will search for products to purchase. "
                    "Your privacy type may change. If you receive personalized pricing after sharing data, or if your privacy type represents your preference for product quality, there is no need for a search decision or privacy cost. The subsequent prompts regarding search decisions will only apply to one scenario."
                    f"You have a privacy value of {self.privacy_cost}. "
                    f"There are {self.num_firms} firms to choose from. "
                    "Please consider whether to share your data. "
                    "If you share your data, you will receive personalized recommendations(your search sequence of products will be sorted from highest value to lowest value), "  # 强调分享数据后个性化推荐的作用！
                    "which will reduce your total search cost.(The total search cost is the product of the number of searches and the search cost. Sharing your data can significantly reduce the number of searches to make a good deal.) "
                    f"If you don't share your data, you won't receive any recommendations, which will increase your total search cost on average.(In the worse case, the total search cost is the product of the number of searches and the search cost, which is {self.num_firms - 1} * {self.search_cost}.) "
                    "Your goal is to maximize your total revenue. "
                    "Your total revenue is the sum of the valuation of the product you purchase, "
                    "minus the price of the product, minus the total search cost, minus the privacy value. "
                    "When deciding whether to share your data, consider the trade-off between the privacy value and the potential reduction in search cost."
                    "If you choose to search more than a certain number, then it's impossible for you to get a positive profit."
                    "When deciding whether to search, consider the trade-off between the search cost and the potential revenue. "
                    f"Given the search cost is {self.search_cost} per search while the net profit from purchasing a product is less than 1, you should be cautious."
                    f"The first search is free, but each subsequent search will cost you {self.search_cost}. "
                    "Please output the detailed logic and thinking process behind any of your decisions in your reason."

        )

    # 新一轮消费者数据还原
    def reset(self):
        self.share = False
        self.purchase_index = -1
        self.total_search_cost = 0
        self.total_revenue = 0
        self.searched_firms = []
        self.temp_memory = {}  # 确保重置为空字典
        
        self.utility = 0.0
        self.purchase_from = None
        self.search_times = 0

    # 计算消费者对公司产品的估值，参数分别是：公司数量，联合分布类型（截断正态分布、独立均匀分布、截断指数分布），分布参数（可选，给出默认参数）
    def generate_valuations(self, num_firms, dist_type='uniform', scale=10, dist_params=None):
        if dist_type == 'normal':
            if dist_params is None:
                dist_params = {'mean': 75, 'std': 20}
            valuations = np.random.normal(dist_params['mean'], dist_params['std'], num_firms)
        elif dist_type == 'uniform':
            if dist_params is None:
                valuations = np.random.uniform(0, 1, num_firms)
                valuations = np.round(valuations, 2)
        elif dist_type == 'exponential':
            if dist_params is None:
                dist_params = {'scale': 1, 'loc': 0}
            scale = scale if scale is not None else 10  # 确保scale不为None
            valuations = np.random.exponential(dist_params['scale'], num_firms) + dist_params['loc']
            valuations = np.clip(valuations, dist_params['loc'], dist_params['loc'] + 2 * scale)
        return valuations

    def generate_quality_consciousness(self, dist_type: str = 'uniform', theta_max: float = 10.0,
                                       dist_params: dict = None) -> float:
        if dist_params is None:
            dist_params = {'type': 'uniform', 'low': 0, 'high': 1}

        # 根据分布类型生成θ值
        if dist_type == 'uniform':
            low = dist_params.get('low')
            if low is None:
                low = 0
            high = dist_params.get('high')
            if high is None:
                high = theta_max
            return round(np.random.uniform(low, high), 2)

        elif dist_type == 'normal':
            mean = dist_params.get('mean')
            if mean is None:
                mean = theta_max / 2
            std = dist_params.get('std')
            if std is None:
                std = theta_max / 6
            sample = np.random.normal(mean, std)
            return round(np.clip(sample, 0, theta_max), 2)

        elif dist_type == 'exponential':
            scale = dist_params.get('scale')
            if scale is None:
                scale = theta_max / 3
            loc = dist_params.get('loc')
            if loc is None:
                loc = 0
            sample = np.random.exponential(scale) + loc
            return round(np.clip(sample, 0, theta_max), 2)

        elif dist_type == 'beta':
            alpha = dist_params.get('alpha')
            if alpha is None:
                alpha = 2
            beta = dist_params.get('beta')
            if beta is None:
                beta = 5
            sample = np.random.beta(alpha, beta) * theta_max
            return round(sample, 2)

        else:
            raise ValueError(f"不支持的分布类型: {dist_type}")
    def _get_model(self, model_name=None):
        """获取指定的模型实例，包含重试机制"""
        if model_name:
            max_retries = 3
            retry_delay = 5  # 重试等待时间（秒）
            for attempt in range(max_retries):
                try:
                    original_config = self.model_config_name
                    self.model_config_name = model_name
                    super().__init__(
                        name=f'consumer_{self.index}',
                        model_config_name=model_name,
                        use_memory=True,
                    )
                    model = self.model
                    self.model_config_name = original_config
                    super().__init__(
                        name=f'consumer_{self.index}',
                        model_config_name=original_config,
                        use_memory=True,
                    )
                    return model
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Consumer {self.index} 模型连接错误: {str(e)}. 将在 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                    else:
                        print(f"Consumer {self.index} 在尝试 {max_retries} 次后放弃连接到模型 {model_name}")
                        raise
        return self.model

    def _add_cot_prompt(self, content: str) -> str:
        if self.enable_cot:
            return f"{content}\n\nThink step by step:\n1. Analyze the current situation\n2. Consider the pros and cons of each option\n3. Make your final decision\n4. Provide a brief reasoning\n\nPlease follow these steps and give your final answer:"
        return content

    # 消费者决定是否分享数据
    def decide_share(self, model_name=None, broadcast_message: str = "", rational=False):
        if rational:
            def integrand(v):
                F_v = (v - self.v_dist['low']) / (self.v_dist['high'] - self.v_dist['low'])
                return F_v - F_v ** self.num_firms
            Δ = quad(integrand, self.r, self.v_dist['high'])[0]
            self.share = (Δ >= self.privacy_cost)
            return self.share
        original_content = (
            f"{self.get_memory_for_prompt()}\n"
            "If you share your data, you will receive personalized recommendations.\n"
            "The personalized recommendations will be sorted by your valuation from high to low.\n"
            "The more firms in the market, the more valuable the recommendations will be.\n"
            f"Now there are {self.num_firms} firms to choose from, which is the maximum number of firms you can search.\n"
            "Do you want to share your data? Please answer 'yes' or 'no' with your extremely brief reason."
            "Please output the detailed logic and thinking process behind any of your decisions."
        )
        enhanced_content = self._add_cot_prompt(original_content)
        prompt = [
            {"role": self.prompt.role, "content": self.prompt.content},
            {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
        ]
        model_use=self._resolve_model_name("decide_share",self.model_config_name)
        response = self._get_model(model_use)(prompt).text
        self.share = "yes" in response.lower()
        self.temp_memory['share'] = self.share
        self.temp_memory['share_reason'] = response
        self.temp_memory['index'] = self.index
        print(f"C{self.index + 1} share decision: {'YES' if self.share else 'NO'}, Reason: {response}")
        return self.share
    #提取模型名
    def _resolve_model_name(self, step: str, explicit_model_name: Optional[str] = None) -> str:
        """
        优先级:
        1) explicit_model_name (来自调用方)(需要改为None)
        2) self.model_names.get(step) (per-consumer override)
        3) self.model_config_name (消费者默认)
        返回最终将传入 self._get_model(...) 的字符串（非 None）。
        """
        if explicit_model_name:
            return explicit_model_name
        if isinstance(self.model_names, dict) and step in self.model_names and self.model_names[step]:
            return self.model_names[step]
        return self.model_config_name

    #定价分享函数
    def decide_share_price(self, model_name=None, broadcast_message: str = "", rational=False):
        if rational:
            pass
        original_content = (
            f"{self.get_memory_for_prompt()}\n"
            "In this scenario, there are no search-related decisions."
            "If you share your data, you will receive personalized pricing based on your needs\n"
            "If you do not share your data, you will be charged the public list prices set by the companies. "
            "After you decide whether to share your data, you will be informed of your consumption type—specifically, your valuation of the product's price. "
            f"Now there are {self.num_firms} firms to choose from, which is the maximum number of firms you can choose to buy from.\n"
            "Do you want to share your data? Please answer 'yes' or 'no' with your extremely brief reason."
        )
        enhanced_content = self._add_cot_prompt(original_content)
        prompt = [
            {"role": self.prompt.role, "content": self.prompt.content},
            {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
        ]
        response = self._get_model(model_name)(prompt).text
        self.share = "yes" in response.lower()
        self.temp_memory['share'] = self.share
        self.temp_memory['share_reason'] = response
        self.temp_memory['index'] = self.index
        print(f"C{self.index + 1} share decision: {'YES' if self.share else 'NO'}, Reason: {response}")
        return self.share
    def decide_share_product_design(self, model_name=None, broadcast_message: str = "", rational=False):
        if rational:
            pass
        original_content = (
            f"{self.get_memory_for_prompt()}\n"
            "In this scenario, there are no search-related decisions.And you don't have the cost of privacy."
            "Your privacy parameter θ represents your level of emphasis on product quality. A higher θ indicates greater emphasis on quality, which can be understood as a stronger tendency to purchase high-quality products."
            f"Your quality-consciousness is {self.quality_consciousness}."
            "If you share your data, you will receive personalized price-quality pairs provided by the company, which are not accessible to other consumers."
            "If you do not share your data, you will receive a 'public' product menu."
            "In this scenario, you must purchase a product."
            "Do you want to share your data? Please answer 'yes' or 'no' with your extremely brief reason."
        )
        enhanced_content = self._add_cot_prompt(original_content)
        prompt = [
            {"role": self.prompt.role, "content": self.prompt.content},
            {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
        ]
        response = self._get_model(model_name)(prompt).text
        self.share = "yes" in response.lower()
        self.temp_memory['share'] = self.share
        self.temp_memory['share_reason'] = response
        self.temp_memory['index'] = self.index
        print(f"C{self.index + 1} share decision: {'YES' if self.share else 'NO'}, Reason: {response}")
        return self.share
    # 消费者决定搜索或购买
    def decide_search(self, platform, model_name=None, broadcast_message: str = "", rational=False):
        if rational:
            market_price = np.mean(platform.firm_prices) if platform.firm_prices else 0.0
            
            if self.share:
                # 共享数据消费者选择最高估值的企业
                max_val = max(self.valuations)
                if max_val > market_price:
                    self.purchase_index = np.argmax(self.valuations)
                    self.purchase_from = self.purchase_index  
                    # 使用实际价格而不是市场价格
                    # actual_price = platform.firm_prices[self.purchase_index] if self.purchase_index < len(platform.firm_prices) else market_price
                    actual_price = platform.firm_prices[self.purchase_index]
                    # 这里到底扣不扣除privacy_cost？非rational状态是扣了的，simulation是没扣的 A:扣
                    self.total_revenue = max_val - actual_price - self.privacy_cost
                    self.utility = self.total_revenue  
                    # 注意：这里本来应该是utility，但为了兼容现有代码结构，暂时保持total_revenue命名
                else:
                    self.purchase_index = -1
                    self.purchase_from = None  
                    # 即使不购买也要扣除privacy_cost（如果分享了数据）
                    self.total_revenue = -self.privacy_cost
                    self.utility = self.total_revenue  
                self.total_search_cost = 0  # 共享数据消费者无需搜索
                self.search_times = 0  # 添加搜索次数记录
                # 添加虚拟搜索记录到 searched_firms
                self.searched_firms.append({
                    'index': np.argmax(self.valuations),
                    'valuation': max_val,
                    'price': platform.firm_prices[np.argmax(self.valuations)]
                })
            else:
                # 非共享数据消费者随机搜索
                searched = []
                search_order = np.random.permutation(self.num_firms)
                search_count = 0  # 搜索计数器
                
                for firm_idx in search_order:
                    search_count += 1  # 每次搜索计数
                    v_i = self.valuations[firm_idx]
                    # p_i = platform.firm_prices[firm_idx] if firm_idx < len(platform.firm_prices) else market_price
                    p_i = platform.firm_prices[firm_idx]
                    net_utility = v_i - p_i

                    # 记录搜索过的公司
                    self.searched_firms.append({
                        'index': firm_idx,
                        'valuation': v_i,
                        'price': p_i
                    })
                    
                    if net_utility >= self.r - market_price:
                        self.purchase_index = firm_idx
                        self.purchase_from = firm_idx  
                        self.total_revenue = net_utility
                        self.utility = net_utility  
                        break
                    searched.append((firm_idx, net_utility))
                else:
                    if searched:
                        max_net_utility = max([net_u for _, net_u in searched])
                        if max_net_utility > 0:
                            self.purchase_index = [idx for idx, net_u in searched if net_u == max_net_utility][0]
                            self.purchase_from = self.purchase_index  
                            self.total_revenue = max_net_utility
                            self.utility = max_net_utility  
                        else:
                            self.purchase_index = -1
                            self.purchase_from = None  
                            self.total_revenue = 0.0
                            self.utility = 0.0  
                    else:
                        self.purchase_index = -1
                        self.purchase_from = None  
                        self.total_revenue = 0.0
                        self.utility = 0.0  
                
                search_cost = max(search_count - 1, 0) * (self.s if hasattr(self, 's') and self.s else platform.search_cost)
                self.total_search_cost = search_cost
                self.total_revenue -= search_cost  # 从总收益中扣除搜索成本
                self.search_times = search_count  # 添加搜索次数记录
            
            self.temp_memory['total_revenue'] = self.total_revenue
            return
        self.total_search_cost = -platform.search_cost
        count = 0
        searched_firms_list = []
        llm_choice = self.num_firms
        decision_sequence = []
        while True:
            # 1. 如果意图是搜索，则执行搜索 (如果还能搜)
            if llm_choice == self.num_firms:
                can_search_more = count < len(platform.search_sequence[self.index])
                if can_search_more:
                    self.total_search_cost += platform.search_cost
                    firm_idx = platform.search_sequence[self.index][count]
                    if firm_idx < len(platform.firm_prices):
                        self.searched_firms.append({
                            'index': firm_idx,
                            'valuation': self.valuations[firm_idx],
                            'price': platform.firm_prices[firm_idx]
                        })
                        count += 1
                        searched_firms_list = [(f['index'], f['valuation'], f['price']) for f in self.searched_firms]
                        self.temp_memory['searched_firms'] = searched_firms_list
                    else:
                        print(f"Warning: F_idx {firm_idx} OOB during search sequence. Stopping search.")
                        # 如果索引越界，强制结束搜索，进入决策（比如离开或基于已搜结果购买）
                        llm_choice = -2 # 使用-2或其他非搜索值触发决策
                        can_search_more = False # 标记不能再搜索

                else:
                    # 如果不能再搜索了 (搜完了或之前遇到问题)
                    # print(f"Debug: Consumer {self.index + 1} cannot search more (count={count}, sequence_len={len(platform.search_sequence[self.index])}). Forcing decision.")
                    llm_choice = -2 # 强制进入决策阶段
                    can_search_more = False

            # --- 决策阶段：每次搜索后（或不能搜索时）进行决策 ---
            # 只有在 self.searched_firms 非空时才有意义做决策
            if not self.searched_firms:
                 print(f"Warning: C{self.index + 1} has not searched any firms yet. Cannot decide. Forcing leave.")
                 llm_choice = -1 # 如果连一家都没搜到（比如序列为空或一开始就出界），则直接离开
                 reason = "No firms searched."
                 decision_sequence.append(f"Step 0: {self.index + 1} forced to leave (no firms searched)")

            else:
                # 准备LLM输入
                profits = [round(f['valuation'] - f['price'], 2) for f in self.searched_firms]
                can_search_more = count < len(platform.search_sequence[self.index]) # 重新检查是否还能搜索

                original_content = (
                    f"{self.share and 'you have decided to share your data' or 'you have decided not to share your data'}"
                    f"You have searched {len(self.searched_firms)} firms. "
                    f"The indices, valuations and prices of these firms are: {searched_firms_list}. **Format: (index, valuation, price)** "
                    f"And their potential profit (valuation - price) is: {profits}. "
                    f"{self.get_memory_for_prompt()}\n"
                    f"So far, your total search cost is {self.total_search_cost}. "
                    # f"Valuations are <= 1. Search cost is {platform.search_cost} per search(the first was free). Be cautious about searching further. "
                    "You are rational: if any searched firm offers positive profit, you won't leave without buying. "
                    "\n\nPlease choose **one** action:"
                    f"\n1. **Purchase:** Choose the index of a firm you already searched (from the list above). "
                    f"\n2. **Search:** Choose {self.num_firms} to search the next firm " + (f"(this will cost {platform.search_cost})." if can_search_more else "(Not possible, all firms searched).") +
                    f"\n3. **Leave:** Choose -1 (stop searching, buy nothing)."
                    "\n\nProvide your choice (only the number) and extremely brief reason."
                    "\nFormat Examples (Strictly follow one):"
                    "\n### Decision:\nI'll **purchase Firm X** (choose number X) because: \n- Reason 1\n- Reason 2"
                    f"\n### Decision:\nI'll **search another firm** (choose number {self.num_firms}) because: \n- Reason 1\n- Reason 2"
                    "\n### Decision:\nI'll **leave** (choose number -1) because: \n- Reason 1\n- Reason 2"
                    "Please output the detailed logic and thinking process behind any of your decisions in your reason."
                )

                enhanced_content = self._add_cot_prompt(original_content)
                prompt = [
                    {"role": self.prompt.role, "content": self.prompt.content},
                    {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
                ]

                # --- LLM 调用和决策逻辑 ---
                max_retries = 3
                retry_count = 0
                reason = ""
                valid_choice = False
                current_decision = -1 # 临时变量存储本轮决策

                while retry_count < max_retries and not valid_choice:
                    try:
                        model_use=self._resolve_model_name("decide_search",self.model_config_name)
                        response = self._get_model(model_use)(prompt).text

                        extracted_choice = None
                        nums = re.findall(r'[-]?\d+', response)
                        # More specific regex: Look for the number directly associated with keywords
                        p_match = re.search(r'(purchase|buy).*?\s*([-]?\d+)', response, re.IGNORECASE | re.DOTALL)
                        s_match = re.search(r'(search|explore).*?\s*(' + str(self.num_firms) + r')', response, re.IGNORECASE | re.DOTALL)
                        l_match = re.search(r'(leave|quit|stop).*?\s*(-1)', response, re.IGNORECASE | re.DOTALL)

                        if p_match:
                            extracted_choice = int(p_match.group(2))
                        elif l_match:
                            extracted_choice = -1
                        elif s_match and can_search_more: # Only accept 'search' if possible
                            extracted_choice = self.num_firms
                        elif nums: # Fallback: check numbers if specific keywords failed
                            for num_str in nums:
                                num = int(num_str)
                                if (num == -1 or
                                    (num == self.num_firms and can_search_more) or
                                    num in [f['index'] for f in self.searched_firms]):
                                    extracted_choice = num
                                    break # Take the first valid number

                        if extracted_choice is None:
                            print(f"Warning C{self.index}: Could not extract valid choice from response. Retrying.")
                            reason = "Error: No choice extracted."
                        elif extracted_choice == self.num_firms:
                            if not can_search_more:
                                print(f"Warning C{self.index}: LLM chose to search, but not possible. Correcting.")
                                # Force decision based on current best - Default logic needed here
                                reason = "Error: Chose search when impossible."
                                extracted_choice = None # Mark as invalid for retry/default
                            else:
                                current_decision = self.num_firms
                                reason = response
                                valid_choice = True
                        elif extracted_choice == -1:
                            current_decision = -1
                            reason = response
                            valid_choice = True
                        elif extracted_choice in [f['index'] for f in self.searched_firms]:
                            current_decision = extracted_choice
                            reason = response
                            valid_choice = True
                        else:
                            print(f"Warning C{self.index}: Extracted choice {extracted_choice} is invalid (not -1, {self.num_firms}, or searched index). Retrying.")
                            reason = f"Error: Invalid choice {extracted_choice}."

                        
                        if not valid_choice:
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"Warning C{self.index}: Max retries reached for search decision. Defaulting based on profit.")
                                best_p = -float('inf')
                                best_c = -1
                                if self.searched_firms: # Ensure list is not empty
                                    for f_data in self.searched_firms:
                                        p = f_data['valuation'] - f_data['price']
                                        if p > best_p:
                                            best_p = p
                                            best_c = f_data['index']
                                    # Default: Buy best positive profit, else leave. Do NOT default to searching more.
                                    current_decision = best_c if best_p > 0 else -1
                                    reason = f"Defaulted after {max_retries} fails. Best Profit: {best_p:.2f} -> Choice: {current_decision}"
                                else: # If somehow no firms were searched despite reaching here
                                     current_decision = -1
                                     reason = f"Defaulted after {max_retries} fails (no firms searched)."
                                valid_choice = True # Exit retry loop after default

                    except Exception as e:
                        print(f"LLM Error C{self.index} decide_search: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"Warning C{self.index}: Max retries reached due to errors. Defaulting based on profit.")
                            # --- Default logic on error (same as above) ---
                            best_p = -float('inf')
                            best_c = -1
                            if self.searched_firms:
                                for f_data in self.searched_firms:
                                    p = f_data['valuation'] - f_data['price']
                                    if p > best_p:
                                        best_p = p
                                        best_c = f_data['index']
                                current_decision = best_c if best_p > 0 else -1
                                reason = f"Defaulted after {max_retries} errors. Best Profit: {best_p:.2f} -> Choice: {current_decision}"
                            else:
                                current_decision = -1
                                reason = f"Defaulted after {max_retries} errors (no firms searched)."
                            valid_choice = True # Exit retry loop after default

                llm_choice = current_decision

                decision_type = "Search Next" if llm_choice == self.num_firms else "Leave Market" if llm_choice == -1 else f"Purchase F{llm_choice}"
                search_costs_so_far = f"SearchCost:{self.total_search_cost:.2f}"
                firms_searched = f"Searched:{len(self.searched_firms)}/{self.num_firms}"

                best_profit = -float('inf')
                best_firm = -1
                for f_data in self.searched_firms:
                    profit = f_data['valuation'] - f_data['price']
                    if profit > best_profit:
                        best_profit = profit
                        best_firm = f_data['index']
                best_info = f"BestProfit:{best_profit:.2f}@F{best_firm}" if best_firm >= 0 else "NoProfits"

                decision_msg = f"C{self.index+1} Search #{count} ({firms_searched}, {search_costs_so_far}, {best_info}) → {decision_type}"
                print(f"\n🔍 {decision_msg}\n   Reason: {reason}")
                decision_sequence.append(decision_msg)

            # --- 检查是否结束循环 ---
            if llm_choice != self.num_firms:
                # 如果决策不是"继续搜索"，则记录最终决定并退出循环
                self.purchase_index = llm_choice # -1 for leave, index for purchase
                self.temp_memory['final_choice'] = self.purchase_index
                self.temp_memory['final_reason'] = reason
                self.temp_memory['decision_sequence'] = decision_sequence  # 保存完整决策序列到记忆

                # 打印决策序列概要
                print(f"\n📊 C{self.index + 1} Search Decision Sequence:")
                for i, decision in enumerate(decision_sequence):
                    print(f"   {i+1}. {decision}")
                print(f"   Final: {'Left Market' if self.purchase_index == -1 else f'Purchased F{self.purchase_index}'}")

                break
            # else: llm_choice is self.num_firms, loop continues to search next

    def decide_purchase(self, platform, model_name=None, broadcast_message: str = ""):
        # 收集所有可购买的公司选项
        firm_options = []
        pers_prices = platform.firm_personalized_prices.get(self.index, {})

        for i in range(self.num_firms):
            # 获取价格 (个性化价格或普通价格)
            price = pers_prices.get(i, platform.firm_prices[i] if i < len(platform.firm_prices) else None)
            if price is None:
                continue

            # 计算估值和利润
            val = self.valuations[i]
            profit = round(val - price, 2)
            firm_options.append({'index': i, 'valuation': val, 'price': price, 'profit': profit})

        # 格式化选项字符串
        opts_str = ", ".join([
            f"(Idx:{f['index']},Val:{f['valuation']},Prc:{f['price']},Prft:{f['profit']})"
            for f in firm_options
        ])

        # 构建提示内容
        original_content = (
            f"{self.get_memory_for_prompt()}\n"
            f"You have the following valuations and prices for all firms: {opts_str}. "
            "The format is (Idx:Firm Index, Val:Your Valuation, Prc:Price Offered, Prft:Your Profit)."
            f"Your historical decisions are: {self.memory}. "
            "Please choose a firm to purchase from or decide not to purchase. "
            "If you want to purchase, please provide the index of the firm. "
            "If you want to leave (stop searching and not purchase), please provide -1. "
            "Please provide your choice (only a number) and extremely brief reason."
            "Format 1 (purchase): (Example which you must strictly follow its format) ### Decision:"
            "I'll **purchase Firm 1** (choose number 1) because:"
            "- ..."
            "- ..."
            "Format 2 (leave): (Example which you must strictly follow its format) ### Decision:"
            "I'll **leave** (choose number -1) because:"
            "- ..."
            "- ..."
        )

        enhanced_content = self._add_cot_prompt(original_content)
        prompt = [
            {"role": self.prompt.role, "content": self.prompt.content},
            {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
        ]


        # 初始化决策变量
        max_retries = 3
        retry_count = 0
        choice = -1
        reason = "Defaulted"
        valid_choice = False

        while retry_count < max_retries and not valid_choice:
            try:
                model_use = self._resolve_model_name("decide_search", self.model_config_name)
                response = self._get_model(model_use)(prompt).text
                # print(f"Debug: Consumer {self.index + 1} has made a choice: {response}.") # 注释掉不必要的打印

                
                extracted_choice = None
                nums = re.findall(r'[-]?\d+', response)
                p_match = re.search(r'(purchase|buy).*? ([-]?\d+)', response, re.IGNORECASE)
                l_match = re.search(r'(leave|quit|stop).*?(-1)', response, re.IGNORECASE)

                if p_match:
                    extracted_choice = int(p_match.group(2))
                elif l_match:
                    extracted_choice = -1
                elif nums:
                    for num_str in nums:
                        num = int(num_str)
                        if num == -1 or num in [f['index'] for f in firm_options]:
                            extracted_choice = num
                            break

                if extracted_choice is None:
                    print("No match found")
                    reason = "Error: No choice."
                elif extracted_choice == -1:
                    choice = -1
                    reason = response
                    valid_choice = True
                elif extracted_choice in [f['index'] for f in firm_options]:
                    choice = extracted_choice
                    reason = response
                    valid_choice = True
                else:
                    print("No match found")  # 或者可以改为 Invalid choice 的打印
                    reason = f"Error: Invalid choice {extracted_choice}."

                if not valid_choice:
                    retry_count += 1
                    if retry_count >= max_retries:  # 默认逻辑
                        print("Max retries reached. Defaulting based on profit.")
                        best_p = -float('inf')
                        best_c = -1

                        for f_opt in firm_options:
                            p = f_opt['profit']
                            if p > best_p:
                                best_p = p
                                best_c = f_opt['index']

                        choice = best_c if best_p > 0 else -1
                        reason = f"Defaulted after fails."
                        valid_choice = True

            except Exception as e:
                print(f"LLM Error:{e}")
                retry_count += 1

                if retry_count >= max_retries:  # 错误后的默认逻辑
                    print("Max retries reached due to errors. Defaulting based on profit.")
                    best_p = -float('inf')
                    best_c = -1

                    for f_opt in firm_options:
                        p = f_opt['profit']
                        if p > best_p:
                            best_p = p
                            best_c = f_opt['index']

                    choice = best_c if best_p > 0 else -1
                    reason = f"Defaulted after errors."
                    valid_choice = True

        self.purchase_index = choice
        self.temp_memory['purchase_choice'] = self.purchase_index
        self.temp_memory['purchase_reason'] = reason

        print(f"Debug: C{self.index + 1} (Pricing) decided: {'Leave' if choice == -1 else f'Purchase F{choice}'}. Reason: {reason}")

    def decide_purchase_price(self, platform, firms, rational=False, model_name=None, broadcast_message: str = ""):
        """情形二专用购买决策：无需搜索，直接基于价格（含个性化价格）决策
        消费者已知所有产品估值和价格，直接选择最优选项
        """
        # 收集所有企业的价格选项（区分个性化/列表价格）
        firm_options = []
        # 从平台获取个性化价格（结构：{consumer_idx: {firm_idx: price}}）
        pers_prices = platform.firm_personalized_prices.get(self.index, {})

        for firm in firms:
            firm_idx = firm.index
            # 1. 确定有效价格（分享数据的消费者可选择最低价格）
            if self.share and firm_idx in pers_prices:
                # 分享数据：对比个性化价格和列表价格，取最低值
                personal_price = pers_prices[firm_idx]
                list_price = firm.price
                effective_price = min(personal_price, list_price)
                price_type = "personalized" if personal_price < list_price else "list"
            else:
                # 未分享数据：仅能看到列表价格
                effective_price = firm.price
                price_type = "list"

            # 2. 计算购买该企业产品的利润（估值 - 有效价格）
            valuation = self.valuations[firm_idx]
            profit = round(valuation - effective_price, 4)

            firm_options.append({
                "index": firm_idx,
                "valuation": valuation,
                "effective_price": effective_price,
                "list_price": firm.price,
                "personal_price": pers_prices.get(firm_idx) if self.share else None,
                "profit": profit,
                "price_type": price_type
            })

        # 3. 构建决策逻辑（统一用 rational 参数）
        if rational:
            # 理性决策：直接选择利润最高的有效选项
            valid_options = [opt for opt in firm_options if opt["profit"] > 0]
            if valid_options:
                best_opt = max(valid_options, key=lambda x: x["profit"])
                choice = best_opt["index"]
                reason = f"Rational choice: Firm {choice} has highest profit ({best_opt['profit']})"
            else:
                choice = -1
                reason = "Rational choice: No firm offers positive profit, do not purchase"
        else:
            # LLM决策：生成英文Prompt
            opts_str = self._format_firm_options(firm_options)
            original_content = self._build_llm_prompt(opts_str)
            if self.enable_cot:
                original_content = self._add_cot_prompt(original_content)

            # 调用LLM并解析结果
            prompt = [
                {"role": self.prompt.role, "content": self.prompt.content},
                {"role": "user", "content": f"{broadcast_message}\n{original_content}"}
            ]
            choice, reason = self._llm_purchase_choice(prompt, firm_options)

        # 4. 记录决策结果
        self.purchase_index = choice
        self.temp_memory.update({
            "purchase_choice": choice,
            "purchase_reason": reason,
            "effective_price": next((opt["effective_price"] for opt in firm_options if opt["index"] == choice), None),
            "profit": next((opt["profit"] for opt in firm_options if opt["index"] == choice), 0.0)
        })

        print(f"消费者{self.index}决策：{'购买企业' + str(choice) if choice != -1 else '不购买'}，"
              f"理由：{reason[:50]}...")
        return choice
    def decide_purchase_product_design(self, platform, firms, model_name=None, broadcast_message: str = "", rational=False):
        # 收集选项
        options = []
        for firm in firms:
            firm_idx = firm.index
            # 公共菜单 (list of (q, p))
            public_menu = platform.firm_public_menu.get(firm_idx, [])
            # 个性化 (q, p) if share
            personal_qp = platform.firm_personalized_products.get(self.index, {}).get(firm_idx) if self.share else None

            # 计算剩余选项
            theta = self.quality_consciousness
            public_surpluses = [(theta * q - p, q, p, "public") for q, p in public_menu if theta * q - p > 0]
            personal_surplus = [(theta * personal_qp[0] - personal_qp[1], personal_qp[0], personal_qp[1], "personal")] if personal_qp and theta * personal_qp[0] - personal_qp[1] > 0 else []

            all_options = public_surpluses + personal_surplus
            options.extend([{"firm_idx": firm_idx, "surplus": s, "q": q, "p": p, "source": source} for s, q, p, source in all_options])

        if rational:
            # 理性：选择剩余最高的
            if options:
                best_opt = max(options, key=lambda x: x["surplus"])
                self.purchase_index = best_opt["firm_idx"]
                self.utility = best_opt["surplus"] - self.privacy_cost if self.share and best_opt["source"] == "personal" else best_opt["surplus"]
                reason = f"Rational: Max surplus {self.utility} from Firm {self.purchase_index}, source {best_opt['source']}"
            else:
                self.purchase_index = -1
                self.utility = -self.privacy_cost if self.share else 0
                reason = "No positive surplus"
        else:
            # LLM 决策
            public_str = "\n".join(
                [f"Public option for Firm {firm.index}: q={q}, p={p}, surplus={theta * q - p}" for firm in firms for
                 q, p in platform.firm_public_menu.get(firm.index, [])])
            personal_str = "\n".join(
                [f"Personalized for Firm {firm.index}: q={qp[0]}, p={qp[1]}, surplus={theta * qp[0] - qp[1]}" for firm
                 in firms for qp in [platform.firm_personalized_products.get(self.index, {}).get(firm.index)] if
                 qp]) if self.share else "No personalized options (you did not share data)."
            content = (f".You are a consumer. You want to purchase a product. Your quality consciousness={theta}. "
                       f"You have {'shared your data' if self.share else 'not shared your data'}. "
                       f"Public menu options (available to all):{public_str}\n"
                       f"Personalized options (if shared):{personal_str}\n"
                       "Decision: 1 for personalized, 2 for public, -1 for no purchase. "
                       "Output format: Decision: X. Reason: your brief reason.")
            enhanced = self._add_cot_prompt(content)
            prompt = [{"role": "user", "content": f"{broadcast_message}\n{enhanced}"}]
            response = self._get_model(model_name)(prompt).text
            # 提取决策
            match = re.match(r"Decision: (\d+|-1)\. Reason: (.*)", response.strip())
            if match:
                decision = int(match.group(1))
                reason = match.group(2)
                if decision == 1 and self.share:
                    # 个性化：假设从选项中选剩余最高的个性化
                    personal_options = [opt for opt in options if opt["source"] == "personal"]
                    best_personal = max(personal_options, key=lambda x: x["surplus"]) if personal_options else None
                    if best_personal:
                        self.purchase_index = best_personal["firm_idx"]
                        self.utility = best_personal["surplus"] - self.privacy_cost
                elif decision == 2:
                    # 公共：选剩余最高的公共
                    public_options = [opt for opt in options if opt["source"] == "public"]
                    best_public = max(public_options, key=lambda x: x["surplus"]) if public_options else None
                    if best_public:
                        self.purchase_index = best_public["firm_idx"]
                        self.utility = best_public["surplus"] - self.privacy_cost if self.share else best_public[
                            "surplus"]
                else:
                    self.purchase_index = -1
                    self.utility = -self.privacy_cost if self.share else 0
            else:
                # 默认不购买如果解析失败
                self.purchase_index = -1
                self.utility = -self.privacy_cost if self.share else 0
                reason = "Parse error: " + response

            self.temp_memory['purchase'] = self.purchase_index
            self.temp_memory['utility'] = self.utility
            self.temp_memory['reason'] = reason
            return self.purchase_index
    def _format_firm_options(self, firm_options):
        """格式化企业选项为字符串，用于LLM Prompt"""
        opts_str = []
        for opt in firm_options:
            price_details = (
                f"Effective Price: {opt['effective_price']} ({opt['price_type']})"
                f" | List Price: {opt['list_price']}"
            )
            if self.share and opt["personal_price"] is not None:
                price_details += f" | Personalized Price: {opt['personal_price']}"
            opts_str.append(
                f"Firm {opt['index']}: Valuation={opt['valuation']}, {price_details}, Profit={opt['profit']}"
            )
        return "\n".join(opts_str)

    def _build_llm_prompt(self, opts_str):
        """构建英文Prompt，明确告知无需搜索、直接决策"""
        return (
            f"You are Consumer {self.index}. You have {'shared your data' if self.share else 'not shared your data'}.\n"
            f"Below are the valuations and prices for all firms:\n{opts_str}\n"
            "You do NOT need to search—you can directly view all prices.\n"
            "Your goal is to maximize your profit (valuation - price) by choosing a firm to purchase from, "
            "or decide not to purchase any product.\n"
            "If all firms offer negative profit, it's better not to purchase.\n"
            "Please output your choice (firm index, or -1 to not purchase) and your reasoning.\n"
            "Format Example:\n"
            "### Decision: Purchase from Firm 2, Reason: Highest profit (0.3)\n"
            "or\n"
            "### Decision: Do not purchase (-1), Reason: All firms have negative profit"
        )

    def _add_cot_prompt(self, content: str) -> str:
        """为LLM添加Chain-of-Thought提示（英文）"""
        if self.enable_cot:
            return f"{content}\n\nThink step by step:\n1. Compare profits across all firms\n2. Identify the firm with the highest positive profit\n3. If no positive profit exists, choose not to purchase\n4. Explain your decision clearly\n\nFinal Answer:"
        return content

    def _llm_purchase_choice(self, prompt, firm_options):
        """LLM决策辅助函数：处理调用与结果解析"""
        max_retries = 3
        retry_count = 0
        choice = -1
        reason = "Default: Do not purchase"
        valid_choices = [opt["index"] for opt in firm_options] + [-1]

        while retry_count < max_retries:
            try:
                response = self._get_model()(prompt).text
                # 提取数字决策（企业编号或-1）
                num_match = re.search(r"(\d+|-\d)", response)
                if num_match:
                    extracted = int(num_match.group(1))
                    if extracted in valid_choices:
                        choice = extracted
                        reason = response
                        break  # 有效决策，退出重试
            except Exception as e:
                print(f"LLM调用错误：{e}")

            retry_count += 1
            reason = f"LLM decision invalid, retrying ({retry_count}/{max_retries})"

        # 最终fallback：选择利润最高的选项
        if choice == -1 and retry_count >= max_retries:
            valid_options = [opt for opt in firm_options if opt["profit"] > 0]
            if valid_options:
                best_opt = max(valid_options, key=lambda x: x["profit"])
                choice = best_opt["index"]
                reason = f"LLM failed, fallback to highest profit firm {choice} (Profit: {best_opt['profit']})"
            else:
                reason = "LLM failed, no positive profit firms, do not purchase"

        return choice, reason
    def calculate_total_revenue_recommendation(self):
        if self.purchase_index == -1:
            self.total_revenue = round(-self.total_search_cost - (self.privacy_cost if self.share else 0), 2)
        else:
            f_data = next((f for f in self.searched_firms if f['index'] == self.purchase_index), None)
            if f_data:
                self.total_revenue = round(f_data['valuation'] - f_data['price'] - self.total_search_cost - (self.privacy_cost if self.share else 0), 2)
                # print(f_data) # 注释掉不必要的打印
            else:
                print(f"Error: C{self.index+1} purchased {self.purchase_index} not found in searched {self.searched_firms}. Rev baseline.")
                self.total_revenue = round(-self.total_search_cost - (self.privacy_cost if self.share else 0), 2)
        self.temp_memory['total_revenue'] = self.total_revenue
        # print(f"Debug: C{self.index + 1} (Recommend) total revenue: {self.total_revenue}.") # 注释掉不必要的打印

    def calculate_total_revenue_pricing(self, platform, firms):
        if self.purchase_index == -1: self.total_revenue = round(-(self.privacy_cost if self.share else 0), 2)
        else:
            pers_prices = platform.firm_personalized_prices.get(self.index, {}); price_paid = pers_prices.get(self.purchase_index, None)
            if price_paid is None: price_paid = platform.firm_prices[self.purchase_index] if self.purchase_index < len(platform.firm_prices) else None
            if price_paid is None:
                print(f"Error: C{self.index+1} purchased F{self.purchase_index} - no price! Rev baseline.") # 保留错误打印
                self.total_revenue = round(-(self.privacy_cost if self.share else 0), 2)
                self.temp_memory['total_revenue'] = self.total_revenue;
                # print(f"Debug: C{self.index + 1} (Pricing) total revenue: {self.total_revenue}.") # 注释掉不必要的打印
                return
            valuation = self.valuations[self.purchase_index]
            self.total_revenue = round(valuation - price_paid - (self.privacy_cost if self.share else 0), 2)
        self.temp_memory['total_revenue'] = self.total_revenue
        # print(f"Debug: C{self.index + 1} (Pricing) total revenue: {self.total_revenue}.") # 注释掉不必要的打印

    def update_memory(self, round_num=None, model_name=None):
        round_number = round_num + 1 if round_num is not None else 1  # 修复None问题
        if not isinstance(self.temp_memory, dict): 
            self.temp_memory = {}
        if 'index' not in self.temp_memory: 
            self.temp_memory['index'] = self.index
        self.history_memory[f"Round{round_number}"] = copy.deepcopy(self.temp_memory)
        if self.memory_distill: 
            self.update_memory_distill(round_num, model_name)

    @property
    def memory(self): return self.history_memory
    @memory.setter
    def memory(self, value):
        if isinstance(value, dict): self.history_memory = value
        else: pass

    def get_memory_for_prompt(self):
        mem = self.history_memory
        if self.memory_distill and self.memory_distill_text: return f"历史总结: {self.memory_distill_text}"
        if not isinstance(mem, dict) or not mem: return "尚无历史决策记录。"
        if self.memory_truncate > 0:
            try: rounds = sorted(mem.keys()); trunc_rounds = rounds[-self.memory_truncate:]; trunc_mem = {k: mem[k] for k in trunc_rounds}; return f"最近 {self.memory_truncate} 轮历史决策: {trunc_mem}"
            except Exception: return f"历史决策: {str(mem)}"
        return f"历史决策: {str(mem)}"

    def update_memory_distill(self, round_num, model_name=None):
        if not self.memory_distill or not self.temp_memory: return
        base_content = f"Previous summary: {self.memory_distill_text}\nCurrent round ({round_num+1}) data: {self.temp_memory}\nSummarize current round only:"
        enhanced_content = self._add_cot_prompt(base_content) if hasattr(self, '_add_cot_prompt') else base_content
        prompt = [{"role": "system", "content": "Please concisely summarize the current round's decisions and results. Use first person (I)."}, {"role": "user", "content": enhanced_content}]
        try:
            response = self._get_model(model_name)(prompt).text
            self.memory_distill_text.append(f"Round {round_num + 1}: {response}")
            if self.memory_truncate > 0 and len(self.memory_distill_text) > self.memory_truncate: self.memory_distill_text = self.memory_distill_text[-self.memory_truncate:]
        except Exception as e: print(f"Consumer {self.index} memory distillation error: {e}")


# 定义公司类型
class Firm(AgentBase):
    def __init__(self, index, method='adaptive', memory_truncate=-1, memory_distill=False, basic_price=50,
                 pricing_mode='adaptive', firm_cost=0, model_config_name="gpt-config", model_names=None,
                 marginal_cost=None, v_dist=None, r_value=None, enable_cot=False):
        self.history_memory = {}
        self.model_config_name = model_config_name
        super().__init__(name=f'firm_{index}', model_config_name=model_config_name, use_memory=True)
        self.index = index
        self.method = method
        self.memory_truncate = memory_truncate
        self.memory_distill = memory_distill
        self.basic_price = basic_price
        self.pricing_mode = pricing_mode
        self.firm_cost = firm_cost
        self.model_names = model_names
        self.price = 0
        self.personalized_prices = {}
        self.num_consumers = 0
        self.num_firms = 0
        self.share_rate_predicted = 0
        self.revenue = 0
        self.profit = 0
        self.temp_memory = {}
        self.memory_distill_text = []
        self.enable_cot = enable_cot
        self.marginal_cost = marginal_cost  # 用于成本 c(q)
        self.public_menu = []  # 公共菜单: list of (q, p)
        self.personalized_products = {}  # {consumer_idx: (q, p)}

        # 理性决策逻辑参数，提供默认值
        self.c = marginal_cost if marginal_cost is not None else firm_cost  # 边际成本，默认等于firm_cost
        self.v_dist = v_dist if v_dist is not None else {'type': 'uniform', 'low': 0, 'high': 1}  # 估值分布参数，默认uniform[0,1]
        self.r = r_value if r_value is not None else 0.8  # 保留值，默认0.8
        
        self.idx = index  
        self.demand = 0.0

        self.prompt = Msg(name="user", role="user",
                          content="You are a firm. You will set the price of your product. "
                                  "Your goal is to maximize your profit. "
                                  "Your profit is the product of the number of consumers who choose to buy your product and the price, minus the cost. "
                                  "Please consider the number of consumers, the share rate of consumers who share data, and your historical decisions."
                                  "Please output the detailed logic and thinking process behind any of your decisions in your reason.")

    # 公司新一轮数据还原
    def reset(self):
        self.price = 0
        self.personalized_prices = {}
        self.share_rate_predicted = 0
        self.revenue = 0
        self.profit = 0
        self.temp_memory = {}
        self.demand = 0.0
        
        self.set_basic_price()

    def set_basic_price(self):
        print(f"[DEBUG] F{self.index} set_basic_price 开始 - 当前basic_price: {self.basic_price:.4f}")
        #print(f"[DEBUG] F{self.index} pricing_mode: {self.pricing_mode}, history_memory存在: {bool(self.history_memory)}")
        
        if self.pricing_mode == 'adaptive' and self.history_memory:
            try:
                rounds = sorted(self.history_memory.keys())
                print(f"[DEBUG] F{self.index} 历史轮次: {rounds}")
                
                if rounds: 
                    last_mem = self.history_memory.get(rounds[-1], {})
                    print(f"[DEBUG] F{self.index} 最后一轮记忆 {rounds[-1]}: {last_mem}")
                    
                if rounds and 'price' in last_mem and isinstance(last_mem['price'], (int, float)):
                    old_basic_price = self.basic_price
                    self.basic_price = last_mem['price']
                    print(f"[DEBUG] F{self.index} basic_price更新: {old_basic_price:.4f} -> {self.basic_price:.4f}")
                else:
                    print(f"[DEBUG] F{self.index} 未找到有效的price记录，保持当前basic_price: {self.basic_price:.4f}")
            except (AttributeError, KeyError, TypeError) as e:
                print(f"警告: Firm {self.index} 设置基础价格时出错: {str(e)}。使用默认基础价格。")
                pass
        else:
            print(f"[DEBUG] F{self.index} 跳过basic_price更新 - 不满足条件")
        
        print(f"[DEBUG] F{self.index} set_basic_price 完成 - 最终basic_price: {self.basic_price:.4f}")

    def get_num_consumers_and_firms(self, platform):
        self.num_consumers=platform.num_consumers; self.num_firms=platform.num_firms
        self.temp_memory['num_consumers']=self.num_consumers; self.temp_memory['num_firms']=self.num_firms

    def form_expectation(self, round_num):
        last_key = f"Round{round_num}"
        if not self.history_memory: # 恢复原始的空记忆处理
            random_value = random.random()
            print(f"Firm {self.index}: 记忆为空，使用随机预期值 {random_value}")
            return random_value
        try:
            if self.method == 'adaptive':
                share_rate = self.history_memory.get(last_key, {}).get('actual_share_rate')
                if share_rate is not None: return share_rate
                else:
                    random_value = random.random()
                    print(f"Firm {self.index}: 找不到Round{round_num}的分享率，使用随机预期值 {random_value}")
                    return random_value
            elif self.method == 'mean':
                rates = [m.get('actual_share_rate') for m in self.history_memory.values() if isinstance(m.get('actual_share_rate'), float)]
                if rates: return np.mean(rates)
                else:
                    random_value = random.random()
                    print(f"Firm {self.index}: 没有任何分享率记录，使用随机预期值 {random_value}")
                    return random_value
            elif self.method == 'perfect':
                 # Perfect expectation needs external injection into temp_memory
                 # This part seems incorrect in the original logic if it reads temp_memory for prediction
                 # Assuming perfect means it gets the current actual rate somehow (handled by runner)
                 # Return a placeholder or rely on external setting
                 print(f"Firm {self.index}: Using 'perfect' expectation method (needs external value).")
                 return self.temp_memory.get('actual_share_rate', 0.5) # Use temp if set, else default
            else:
                random_value = random.random()
                print(f"Firm {self.index}: 未知预期方法 {self.method}，使用随机预期值 {random_value}")
                return random_value
        except Exception as e:
            random_value = random.random()
            print(f"Firm {self.index}: 计算预期时出错 {str(e)}，使用随机预期值 {random_value}")
            return random_value

    def update_expectation(self, round_num):
        if not isinstance(self.temp_memory, dict): self.temp_memory = {}
        self.share_rate_predicted = self.form_expectation(round_num)
        self.temp_memory['share_rate_predicted'] = [self.share_rate_predicted, self.method]

    def set_price(self, model_name=None, broadcast_message: str = "", rational=False):
        if rational:
            σ = self.share_rate_predicted
            print(f"Firm {self.index}: Setting price rationally with σ={σ:.2f}, c={self.c:.2f}, r={self.r:.2f}, v_dist={self.v_dist}")
            n = self.num_firms
            print(f"Firm {self.index}: Number of firms n={n}")
            # market_price是通过平台传入的参数
            # 这里我们使用基本价格作为初始市场价格估计
            market_price = self.basic_price  
            
            r = self.r
            v_low = self.v_dist['low']
            v_high = self.v_dist['high']
            
            try:
                F_r = (r - v_low) / (v_high - v_low)
                f_r = 1 / (v_high - v_low)
                denominator = n * (F_r ** (n - 1)) * f_r
                condition = (r - self.c) > (1 - F_r ** n) / denominator if denominator > 0 else False
            except:
                condition = False
                
            if not condition:
                print(f"Firm {self.index}: Condition not satisfied, setting price to fallback.")
                self.price = min(r, v_high * 0.999)
                return
                
            def d_profit_d_pi(p_i, p):
                F_pi = (p_i - v_low) / (v_high - v_low)
                F_pi = np.clip(F_pi, 0, 1)
                q_s = (1 - F_pi ** n) / n
                dq_s = -F_pi ** (n - 1) / (v_high - v_low) if F_pi < 1 else 0
                
                q_ns = self._non_shared_demand(p_i, p, r, n)
                dq_ns = self._deriv_non_shared(p_i, p, r, n)
                
                Q = σ * q_s + (1 - σ) * q_ns
                dQ = σ * dq_s + (1 - σ) * dq_ns
                return Q + (p_i - self.c) * dQ
                
            def foc(p):
                return d_profit_d_pi(p, p)
                
            try:
                result = root_scalar(foc, method='bisect', bracket=[self.c, r], maxiter=1000)
                self.price = result.root
            except Exception as e:
                print(f"Firm {self.index}: Failed to solve for equilibrium price, using fallback. Error: {e}")
                self.price = np.clip((self.c + r) / 2, self.c, r)
            return

        # 如果不是理性定价，则使用LLM进行定价
        original_content = (
            "Your goal is to maximize your total profit. "
            f"Your cost is: {self.firm_cost} per product. "
            "The consumers' valuations for every firm's product are independently uniformly distributed between 0 and 1. "
            f"There are {self.num_firms} firms in the market, including you. "
            "The more firms in the market, the more competition you will face. "
            f"Your predicted share rate for this round is: {self.share_rate_predicted}. "
            "NOTE: THE MORE CONSUMERS SHARE DATA, THE MORE LIKELY THEY ARE TO RECEIVE PERSONALIZED RECOMMENDATIONS"
            "(EVERY FIRM HAS EQUAL CHANCE TO BE RECOMMENDED). "
            "THE MORE CONSUMERS RECEIVE PERSONALIZED RECOMMENDATIONS, THE LESS LIKELY THEY ARE TO SEARCH FOR OTHER FIRMS"
            "(IF OTHER FIRMS EXIST). "
            "THE LESS LIKELY THEY ARE TO SEARCH FOR OTHER FIRMS, THE MORE LIKELY THEY ARE TO BUY YOUR PRODUCT. "
            "THE MORE LIKELY THEY ARE TO BUY YOUR PRODUCT, THE MORE LIKELY YOU ARE TO SET A HIGHER PRICE. "
            "The more competition you face, the more valuable personalized recommendations become, "
            "as they help consumers find the best product more efficiently. "
            "With more firms, consumers are more likely to rely on personalized recommendations, "
            "reducing their search efforts and increasing their likelihood of purchasing your product. "
            f"Your historical decisions are: {self.get_memory_for_prompt()}\n"
            f"Your basic price is: {self.basic_price}. "
            "Please provide a new price for your product and extremely brief reason. "
            "Format: 'Change price: xxx, reason: xxx'."
            "Note: If you want to keep the basic price, then set the Change price to 0."
            "If you want to increase the price, then set the Change price to a positive number."
            "If you want to decrease the price, then set the Change price to a negative number."
            "Be sure to keep the price positive and round to two decimal places.(Like 0.00)"
        )

        enhanced_content = self._add_cot_prompt(original_content)
        prompt = [
            {"role": self.prompt.role, "content": self.prompt.content},
            {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
        ]


        # 初始化变量
        max_retries = 3
        retry_count = 0
        success = False
        response_text = "Error"
        llm_price = self.basic_price

        # 尝试获取LLM定价决策
        while retry_count < max_retries and not success:
            try:
                # 获取LLM响应
                response = self._get_model(model_name)(prompt).text
                response_text = response

                # 尝试从响应中提取价格变化
                match = re.search(r"(?:Change price|New price|Price):\s*([-+]?\d+\.?\d*)", response_text, re.IGNORECASE)

                if match:
                    # 如果找到匹配，计算新价格
                    change = round(float(match.group(1)), 4)
                    calculated_price = self.basic_price + change

                    if calculated_price >= 0:
                        llm_price = calculated_price
                        success = True
                    else:
                        response_text = "Error: Resulting price negative."
                else:
                    # 回退策略：直接查找数字
                    nums = re.findall(r"\d+\.?\d*", response_text)

                    if nums:
                        extracted_price = round(float(nums[0]), 4) 

                        if extracted_price >= 0:
                            llm_price = extracted_price
                            success = True
                        else:
                            response_text = "Error: Fallback neg price."
                    else:
                        response_text = "Error: No price/change found."

            except Exception as e:
                print(f"LLM Error F{self.index} set_price: {e}")
                response_text = f"Error: {e}"

            # 处理重试逻辑
            if not success:
                retry_count += 1

            if retry_count >= max_retries:
                print(f"F{self.index} set_price failed. Default: {llm_price:.4f}")
                response_text = "Error: Max retries"
                success = True

        # 设置最终价格并保存到记忆
        self.price = round(llm_price, 4)  # 改为4位小数
        print(f"F{self.index} 设置价格: {self.price:.4f} (basic: {self.basic_price:.4f}, change: {self.price - self.basic_price:+.4f}), Reason: {response_text}")

        # 确保价格非负
        if self.price < 0:
            self.price = 0

        # 更新临时记忆
        self.temp_memory['price'] = self.price
        self.temp_memory['price_reason'] = response_text
        self.temp_memory['index'] = self.index

    def _non_shared_demand(self, p_i, p, r, n):
        v_low = self.v_dist['low']
        v_high = self.v_dist['high']
        F_r = (r - v_low) / (v_high - v_low)
        F_term = (r - p + p_i - v_low) / (v_high - v_low)
        F_term = np.clip(F_term, 0, 1)
        numerator = (1 - F_term) * (1 - F_r ** n)
        denominator = n * (1 - F_r) if (1 - F_r) > 0 else 1e-10
        term1 = numerator / denominator
        lower = p_i
        upper = r - p + p_i
        if lower >= upper:
            term2 = 0.0
        else:
            def integrand(v_i):
                offset = v_i - p_i + p
                F_offset = (offset - v_low) / (v_high - v_low)
                F_offset = np.clip(F_offset, 0, 1)
                return F_offset ** (n - 1) * (1 / (v_high - v_low))
            term2, _ = quad(integrand, lower, upper)
        return term1 + term2

    def _deriv_non_shared(self, p_i, p, r, n):
        v_low = self.v_dist['low']
        v_high = self.v_dist['high']
        v_span = v_high - v_low
        F_r = (r - v_low) / v_span
        F_p = (p - v_low) / v_span
        f_density = 1 / v_span
        d_term1 = - (1 / n) * f_density * (1 - F_r ** n) / (1 - F_r) if (1 - F_r) > 0 else 0
        def integrand(v_i):
            offset = v_i - p_i + p
            F_offset = (offset - v_low) / v_span
            F_offset = np.clip(F_offset, 0, 1)
            if F_offset == 0 or F_offset == 1:
                return 0
            return (n - 1) * F_offset ** (n - 2) * f_density * f_density
        lower = max(p_i, v_low)
        upper = min(r - p + p_i, v_high)
        if lower >= upper:
            integral_term = 0.0
        else:
            integral_term, _ = quad(integrand, lower, upper)
        F_r_term = F_r ** (n - 1) * f_density if (r - p + p_i >= v_low and r - p + p_i <= v_high) else 0
        F_p_term = F_p ** (n - 1) * f_density if (p_i >= v_low and p_i <= v_high) else 0
        boundary_term = F_r_term - F_p_term
        d_term2 = boundary_term - integral_term
        return d_term1 + d_term2

    def set_price_pricing(self, model_name=None, broadcast_message: str = "",rational=False):
        if rational:
            pass
        original_content = (
            "Your goal is to maximize your total profit. "
            f"Your cost is: {self.firm_cost} per product. "
            "The consumers' valuations for every firm's product are independently uniformly distributed between 0 and 1. "
            f"There are {self.num_firms} firms in the market, including you. "
            "The more firms in the market, the more competition you will face. "
            f"Your predicted share rate for this round is: {self.share_rate_predicted}. "
            "NOTE: FOR CONSUMERS WHO SHARE THEIR DATA(THEIR VALUE FOR ALL PRODUCTS IN THE MARKET), YOU CAN SET A PERSONALIZED DISCOUNT FOR THEM."
            "BEFORE THAT, YOU HAVE TO SET AN ORIGINAL PRICE FOR CONSUMERS WHO REFUSE TO SHARE THEIR DATA."
            "THE MORE CONSUMERS SHARE DATA, THE MORE CONSUMER WILL GET PERSONALIZED DISCOUNTS, SO THE ORIGINAL PRICE IS RELATED TO SHARE RATE."
            f"Your historical decisions are: {self.get_memory_for_prompt()}\n"
            f"Your basic price is: {self.basic_price}. "
            "Please provide a new price(list price) for your product and extremely brief reason. "
            "Format: 'Change price: xxx, reason: xxx'."
            "Note: If you want to keep the basic price, then set the Change price to 0."
            "If you want to increase the price, then set the Change price to a positive number."
            "If you want to decrease the price, then set the Change price to a negative number."
            "Be sure to keep the price positive and round to two decimal places.(Like 0.00)"
        )
        enhanced_content = self._add_cot_prompt(original_content)
        prompt = [
            {"role": self.prompt.role, "content": self.prompt.content},
            {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
        ]
        max_retries=3; retry_count=0; success=False; response_text="Error"; llm_price=self.basic_price
        while retry_count < max_retries and not success:
            try:
                response = self._get_model(model_name)(prompt).text
                response_text = response
                match = re.search(r"(?:Change price|List price|Price):\s*([-+]?\d+\.?\d*)", response_text, re.IGNORECASE)
                if match:
                    change = round(float(match.group(1)), 4); calculated_price = self.basic_price + change  # 改为4位小数
                    if calculated_price >= 0: llm_price = calculated_price; success = True
                    else: response_text = "Error: Resulting price negative."
                else: # Fallback
                    nums = re.findall(r"\d+\.?\d*", response_text)
                    if nums:
                        extracted_price = round(float(nums[0]), 4)  # 改为4位小数
                        if extracted_price >= 0:
                            llm_price = extracted_price
                            success = True
                        else:
                            response_text = "Error: Fallback neg price."
                    else: # This corresponds to 'if nums:'
                        response_text = "Error: No list price/change found."
            except Exception as e:
                print(f"LLM Error F{self.index} set_price_pricing: {e}"); response_text = f"Error: {e}"
            if not success: retry_count += 1
            if retry_count >= max_retries:
                print(f"F{self.index} set_price_pricing failed. Default: {llm_price:.4f}")
                response_text = "Error: Max retries"
                success=True
        self.price = round(llm_price, 4)  # 改为4位小数
        print(f"####  The list price is: {self.price:.4f}")
        if self.price < 0: self.price = 0 # 保留价格非负检查
        self.temp_memory['price'] = self.price
        self.temp_memory['price_reason'] = response_text
        self.temp_memory['index'] = self.index

    def _get_model(self, model_name=None):
        if model_name:
            max_retries = 3; retry_delay = 5
            for attempt in range(max_retries):
                try:
                    original_config = self.model_config_name; self.model_config_name = model_name
                    super().__init__(name=f'firm_{self.index}', model_config_name=model_name, use_memory=True)
                    model = self.model; self.model_config_name = original_config
                    super().__init__(name=f'firm_{self.index}', model_config_name=original_config, use_memory=True)
                    return model
                except Exception as e:
                    if attempt < max_retries - 1: print(f"F{self.index} connect error: {e}. Retry..."); time.sleep(retry_delay); retry_delay *= 1.5
                    else: print(f"F{self.index} give up model {model_name}"); raise
        return self.model

    def _add_cot_prompt(self, content: str) -> str:
        if self.enable_cot:
            return f"{content}\n\nThink step by step:\n1. Analyze the current market situation\n2. Consider competitors' strategies\n3. Evaluate pricing impact on profits\n4. Make your final pricing decision\n5. Provide a brief reasoning\n\nPlease follow these steps and give your final answer:"
        return content

    def set_personalized_price(self, platform, consumer_index, model_name=None, broadcast_message: str = "",rational=False):
        # 获取消费者对产品的估值
        valuations, valuation = platform.get_consumer_valuation(consumer_index, self.index)

        if valuation is None:
            print(f"Debug: No valuation for C{consumer_index}/F{self.index}.")
            return

        # 转换所有估值为字符串展示
        all_vals_str = str(valuations)

        # 构建个性化定价的提示内容
        original_content = (
            f"Consumer {consumer_index} has shared his/her data. "
            f"His/her valuation for all product is: {all_vals_str}. " 
            f"Especially, his/her valuation for your product is: {valuation}"
            "NOTE: EACH FIRM'S PRODUCT HAS THE **SAME** EXPECTED VALUATION FOR ALL CONSUMERS, "
            "SO THE VALUATION'S DIFFERENCE BETWEEN PRODUCTS IS ALL ABOUT PERSONAL PREFERENCE "
            "(WHICH IS TOTALLY RANDOM AND UNPREDICTABLE)."
            "YOU HAVE TO CONSIDER THIS TO DESIGN A COMPETITIVE AND BENEFICIAL PRICE FOR THIS CONSUMER."
            f"Your historical decisions are: {self.get_memory_for_prompt()}\n"
            f"Your list price is: {self.price}. "
            "Please provide a personalized price for this consumer (not higher than the list price) "
            "and extremely brief reason. "
            "Format: 'Change price: xxx, reason: xxx'."
            "Note: If you want to keep the list price, then set the Change price to 0."
            "If you want to increase the price, then set the Change price to a positive number."
            "(However, the personalized price should not be higher than the list price, "
            "because the consumer can always buy the product at the list price.)"
            "If you want to decrease the price, then set the Change price to a negative number."
            "Be sure to keep the price positive and round to two decimal places.(Like 0.00)"
        )

        enhanced_content = self._add_cot_prompt(original_content)
        prompt = [
            {"role": self.prompt.role, "content": self.prompt.content},
            {"role": "user", "content": f"{broadcast_message}\n{enhanced_content}"}
        ]

        # 初始化变量
        max_retries = 3
        retry_count = 0
        success = False
        response_text = "Error"
        personalized_price = self.price

        # 尝试获取LLM个性化定价决策
        while retry_count < max_retries and not success:
            try:
                # 获取LLM响应
                response = self._get_model(model_name)(prompt).text
                response_text = response

                # 尝试从响应中提取价格变化
                match = re.search(
                    r"(?:Change price|Personalized price|Price)\s*(?:for\s*Consumer\s*\d+)?:\s*([-+]?\d+\.?\d*)",
                    response_text,
                    re.IGNORECASE
                )

                if match:
                    # 如果找到匹配，计算新价格
                    change = round(float(match.group(1)), 4)  # 改为4位小数
                    calculated_price = self.price + change

                    if calculated_price >= 0:
                        personalized_price = calculated_price
                        success = True
                    else:
                        response_text = "Error: Resulting price negative."
                else:
                    # 回退策略：直接查找数字
                    nums = re.findall(r"\d+\.?\d*", response_text)

                    if nums:
                        potential_price = round(float(nums[-1]), 4)  # 改为4位小数，提取最后一个数字

                        if potential_price >= 0:
                            personalized_price = potential_price
                            success = True
                        else:
                            response_text = "Error: Fallback neg price."
                    else:
                        response_text = "Error: No pers price/change found."

            except Exception as e:
                print(f"LLM Error F{self.index} set_pers_price C{consumer_index}: {e}")
                response_text = f"Error: {e}"

            # 处理重试逻辑
            if not success:
                retry_count += 1

            if retry_count >= max_retries:
                print(f"F{self.index} set_pers_price failed C{consumer_index}. Default: {personalized_price:.4f}")
                response_text = "Error: Max retries"
                success = True

        # 设置个性化价格并保存
        self.personalized_prices[consumer_index] = round(personalized_price, 4)  # 改为4位小数

        # 确保价格非负
        if self.personalized_prices[consumer_index] < 0:
            self.personalized_prices[consumer_index] = 0

        # 打印最终价格
        print(
            f"####  The price is: {self.personalized_prices[consumer_index]:.4f} "
            f"for Consumer {consumer_index}"
        )

        # 更新临时记忆
        if 'personalized_decisions' not in self.temp_memory:
            self.temp_memory['personalized_decisions'] = {}

        self.temp_memory['personalized_decisions'][consumer_index] = {
            'price': self.personalized_prices[consumer_index],
            'reason': response_text
        }

    def cost_function(self, q):
        return q ** 2 / 2  # 凸成本函数，c(q) = q^2 / 2
    def set_public_menu(self, platform, rational=False, model_name=None, broadcast_message: str = ""):
        if rational:
            # 理性：优化公共菜单（e.g., 2-3 个 (q, p) 选项覆盖分布）
            # 示例：低 q 低 p，中 q 中 p，高 q 高 p
            self.public_menu = [
                (1.0, 0.5),  # 低质量
                (5.0, 3.0),  # 中
                (10.0, 8.0)  # 高
            ]
            reason = "Rational public menu covering distribution"
        else:
            # LLM
            content = (
                "You need to set up standard products that will be visible to all users. You have already obtained the data of consumers who shared their data, and then set up personalized products for each of them respectively. Now you need to set the price-quality pairs for the standard products."
                "Your profit equals the selling price of the products sold minus the cost."
                "q means quality,p means price."
                f"Set public menu: list of (q, p) pairs. Cost c(q) = q^2/2."
                "Output format: Menu: [(q1,p1)]. Reason: brief reason.")
            enhanced = self._add_cot_prompt(content)
            prompt = [{"role": "user", "content": f"{broadcast_message}\n{enhanced}"}]
            response = self._get_model(model_name)(prompt).text
            # 解析 response 为 list of tuples
            self.public_menu = [(float(q), float(p)) for q, p in re.findall(r'\((\d+\.\d+), (\d+\.\d+)\)', response)]
            reason = response

        self.temp_memory['public_menu'] = self.public_menu
        self.temp_memory['public_menu_reason'] = reason
        print(f"Firm {self.index} public menu: {self.public_menu}")
        return self.public_menu
    def set_personalized_product(self, platform, consumer_idx, rational=False, model_name=None, broadcast_message: str = ""):
        theta = platform.get_consumer_quality_consciousness(consumer_idx)  # θ
        if rational:
            # 理性：优化 q* = argmax θq - c(q), p = θq - epsilon (提取剩余)
            q_opt = theta / 2  # 从 c'(q) = θ 求解 (c' = q = θ)
            p_opt = theta * q_opt - 0.01  # 几乎全剩余
            reason = "Rational personalized: Optimize q for θ, extract surplus"
        else:
            # LLM
            content = (
                f"Your standard product's price-quality pair is:{self.public_menu[0]}."
                "q means quality,p means price."
                f"Set personalized (q, p) for consumer {consumer_idx}, whose quality-consciousness={theta},which represents the consumer's level of emphasis on product quality. The higher this value is, the more inclined the consumer is to purchase high-quality products. Cost c(q)=q^2/2."
                "Format: (q, p).")
            enhanced = self._add_cot_prompt(content)
            prompt = [{"role": "user", "content": f"{broadcast_message}\n{enhanced}"}]
            response = self._get_model(model_name)(prompt).text
            q_opt, p_opt = map(float, re.match(r'\((\d+\.\d+), (\d+\.\d+)\)', response).groups())
            reason = response

        self.personalized_products[consumer_idx] = (q_opt, p_opt)
        self.temp_memory[f'personal_{consumer_idx}'] = (q_opt, p_opt)
        self.temp_memory[f'personal_reason_{consumer_idx}'] = reason
        print(f"Firm {self.index} personalized for C{consumer_idx}: q={q_opt}, p={p_opt}")
        return (q_opt, p_opt)
    def get_revenue(self, platform, share_ratio: float, actual_sales: Dict[int, float]):
        num_sales = len(actual_sales); total_revenue_calculated = sum(actual_sales.values())
        total_cost = self.firm_cost * num_sales
        self.revenue = round(total_revenue_calculated, 4)
        self.profit = round(self.revenue - total_cost, 4)
        print(f"Debug: F{self.index} Sales: {num_sales}. Rev: {self.revenue:.4f}, Cost: {total_cost:.4f}, Profit: {self.profit:.4f}.")
        self.temp_memory['actual_share_rate'] = round(share_ratio, 4)
        self.temp_memory['sale_num'] = num_sales
        self.temp_memory['revenue'] = self.revenue
        self.temp_memory['profit'] = self.profit

    def update_memory(self, round_num, model_name=None):
        if not isinstance(self.temp_memory, dict): self.temp_memory = {}
        if 'index' not in self.temp_memory: self.temp_memory['index'] = self.index
        self.history_memory["Round" + str(round_num + 1)] = copy.deepcopy(self.temp_memory)
        if self.memory_distill: self.update_memory_distill(round_num, model_name)

    def get_memory_for_prompt(self):
        mem = self.history_memory
        if self.memory_distill and self.memory_distill_text: return f"历史总结: {self.memory_distill_text}"
        if not isinstance(mem, dict) or not mem: return "尚无历史决策记录。"
        if self.memory_truncate > 0:
            try: rounds = sorted(mem.keys()); trunc_rounds = rounds[-self.memory_truncate:]; trunc_mem = {k: mem[k] for k in trunc_rounds}; return f"最近 {self.memory_truncate} 轮历史决策: {trunc_mem}"
            except Exception: return f"历史决策: {str(mem)}"
        return f"历史决策: {str(mem)}"

    def update_memory_distill(self, round_num, model_name=None):
        if not self.memory_distill or not self.temp_memory: return
        base_content = f"Previous summary: {self.memory_distill_text}\nCurrent round ({round_num+1}) data: {self.temp_memory}\nSummarize current round only:"
        enhanced_content = self._add_cot_prompt(base_content) if hasattr(self, '_add_cot_prompt') else base_content
        prompt = [{"role": "system", "content": "Please concisely summarize the current round's decisions and results. Use first person (I)."}, {"role": "user", "content": enhanced_content}]
        try:
            response = self._get_model(model_name)(prompt).text
            self.memory_distill_text.append(f"Round {round_num + 1}: {response}")
            if self.memory_truncate > 0 and len(self.memory_distill_text) > self.memory_truncate: self.memory_distill_text = self.memory_distill_text[-self.memory_truncate:]
        except Exception as e:
            print(f"Firm {self.index} memory distillation error: {e}")

    @property
    def memory(self): return self.history_memory
    @memory.setter
    def memory(self, value):
        if isinstance(value, dict): self.history_memory = value
        else: pass


# 定义平台类型
class Platform(AgentBase):
    def __init__(self, search_cost=0.0, memory_truncate=-1, memory_distill=False, model_config_name="gpt-config",
                 model_names=None):
        self.history_memory = []
        self.model_config_name = model_config_name
        super().__init__(name='platform', model_config_name=model_config_name, use_memory=True)
        self.search_cost = search_cost;
        self.memory_truncate = memory_truncate;
        self.memory_distill = memory_distill
        self.model_names = model_names;
        self.num_consumers = 0;
        self.num_firms = 0
        self.consumer_valuations = {};
        self.consumer_quality_consciousness={};
        self.firm_public_menu = {};
        self.firm_personalized_products = {};#用于保存个性化产品菜单
        self.all_consumer_quality_consciousness={};
        self.firm_prices = [];
        self.firm_personalized_prices = {}
        self.search_sequence = {};
        self.consumer_purchase_behavior = [];
        self.firm_sales = {}
        self.consumer_surplus = 0;
        self.firm_surplus = 0;
        self.temp_memory = {};
        self.memory_distill_text = []
        self.prompt = Msg(name="system", role="system", content="Platform.")

    def _get_model(self, model_name=None):
        if model_name:
            max_retries = 3;
            retry_delay = 5
            for attempt in range(max_retries):
                try:
                    original_config = self.model_config_name;
                    self.model_config_name = model_name
                    super().__init__(name='platform', model_config_name=model_name, use_memory=True)
                    model = self.model;
                    self.model_config_name = original_config
                    super().__init__(name='platform', model_config_name=original_config, use_memory=True)
                    return model
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Platform model err:{e}. Retry..."); time.sleep(retry_delay); retry_delay *= 1.5
                    else:
                        print(f"Platform give up model {model_name}"); raise
        return self.model

    def reset(self):
        self.consumer_valuations = {};
        self.firm_prices = [];
        self.firm_personalized_prices = {}
        self.search_sequence = {};
        self.consumer_purchase_behavior = [];
        self.firm_sales = {}
        self.consumer_surplus = 0;
        self.firm_surplus = 0;
        self.temp_memory = {}

    def get_num_consumers(self, consumer_list):
        self.num_consumers = len(consumer_list); self.temp_memory['num_consumers'] = self.num_consumers

    def get_num_firms(self, firm_list):
        self.num_firms = len(firm_list); self.temp_memory['num_firms'] = self.num_firms

    def get_consumer_valuations(self, consumer_list):
        self.consumer_valuations = {c.index: c.valuations for c in consumer_list if c.share}; self.temp_memory[
            'consumer_valuations_shared'] = copy.deepcopy(self.consumer_valuations)

    def get_all_consumer_valuations(self, consumer_list):
        self.consumer_valuations = {c.index: c.valuations for c in consumer_list}; self.temp_memory[
            'consumer_valuations_all'] = copy.deepcopy(self.consumer_valuations)

    def get_consumer_valuation(self, c_idx, f_idx):
        vals = self.consumer_valuations.get(c_idx); return (vals, vals[f_idx]) if vals is not None and f_idx < len(
            vals) else (None, None)
    def get_consumers_quality_consciousness(self,consumer_list):
        self.consumer_quality_consciousness = {c.index: c.quality_consciousness for c in consumer_list if c.share}; self.temp_memory[
            'consumer_quality_consciousness_shared'] = copy.deepcopy(self.consumer_quality_consciousness)
    def get_all_consumers_quality_consciousness(self,consumer_list):
        self.all_consumer_quality_consciousness={c.index: c.quality_consciousness for c in consumer_list}; self.temp_memory[
            'consumer_quality_consciousness_all'] = copy.deepcopy(self.consumer_quality_consciousness)
    def get_consumer_quality_consciousness(self, c_idx):
        return self.consumer_quality_consciousness.get(c_idx, None)

    def get_firm_public_menu(self, firms):
        self.firm_public_menu = {f.index: f.public_menu for f in firms}

    def get_firm_personalized_products(self, firms, consumers):
        self.firm_personalized_products = {}
        for c in consumers:
            if c.share:
                self.firm_personalized_products[c.index] = {f.index: f.personalized_products.get(c.index) for f in firms}
    def get_firm_personalized_prices(self, firm_list, consumer_list):
        self.firm_personalized_prices = {}
        for c in consumer_list:
            if c.share: prices = {f.index: f.personalized_prices.get(c.index) for f in firm_list if
                                  f.personalized_prices.get(c.index) is not None};
            if c.share and prices: self.firm_personalized_prices[c.index] = prices
        self.temp_memory['firm_personalized_prices'] = copy.deepcopy(self.firm_personalized_prices)

    def get_all_consumer_quality_consciousness(self, consumers):
        self.consumer_quality_consciousness = {c.index: c.quality_consciousness for c in consumers}

    def get_firm_public_menu(self, firms):
        self.firm_public_menu = {f.index: f.public_menu for f in firms}

    def get_firm_personalized_products(self, firms, consumers):
        self.firm_personalized_products = {}
        for c in consumers:
            if c.share:
                self.firm_personalized_products[c.index] = {f.index: f.personalized_products.get(c.index) for f in
                                                            firms}
    def generate_search_sequence(self, consumer_list):
        self.search_sequence = {c.index: (
            np.argsort(c.valuations)[::-1].tolist() if c.share else np.random.permutation(self.num_firms).tolist()) for
                                c in consumer_list}; self.temp_memory['search_sequences'] = copy.deepcopy(
            self.search_sequence)

    def get_consumer_purchase_behavior_recommendation(self, consumer_list):
        self.consumer_purchase_behavior = []
        for c in consumer_list:
            if c.purchase_index != -1: f_data = next((f for f in c.searched_firms if f['index'] == c.purchase_index),
                                                     None);
            if c.purchase_index != -1 and f_data: self.consumer_purchase_behavior.append(
                {'consumer': c.index, 'firm': c.purchase_index, 'price': f_data['price'],
                 'valuation': f_data['valuation']})
        self.temp_memory['purchase_behavior'] = copy.deepcopy(self.consumer_purchase_behavior)

    def get_consumer_purchase_behavior_pricing(self, consumer_list, firm_list):
        self.consumer_purchase_behavior = []
        for c in consumer_list:
            if c.purchase_index != -1: price_paid = self.firm_personalized_prices.get(c.index, {}).get(
                c.purchase_index);
            if c.purchase_index != -1 and price_paid is None: price_paid = self.firm_prices[
                c.purchase_index] if c.purchase_index < len(self.firm_prices) else None
            if c.purchase_index != -1 and price_paid is not None: self.consumer_purchase_behavior.append(
                {'consumer': c.index, 'firm': c.purchase_index, 'price': price_paid,
                 'valuation': c.valuations[c.purchase_index], 'shared_data': c.share})
        self.temp_memory['purchase_behavior'] = copy.deepcopy(self.consumer_purchase_behavior)

    # 修改: calculate_sales 只计算和存储销售数据
    def calculate_sales(self, firm_list):
        """平台仅计算和记录销售数据，利润计算由运行脚本协调 firm.get_revenue 完成。"""
        self.firm_sales = {f.index: {'sale_num': 0, 'sales_details': {}} for f in firm_list}
        for p in self.consumer_purchase_behavior:
            f_idx, c_idx, price = p['firm'], p['consumer'], p['price']
            if f_idx in self.firm_sales:
                self.firm_sales[f_idx]['sale_num'] += 1
                self.firm_sales[f_idx]['sales_details'][c_idx] = price
        self.temp_memory['firm_sales'] = copy.deepcopy(self.firm_sales)

    def get_firm_prices(self, firm_list):
        self.firm_prices = [f.price for f in firm_list]; self.temp_memory['firm_list_prices'] = copy.deepcopy(
            self.firm_prices)

    def calculate_surplus(self, consumer_list, firm_list):
        # 消费者剩余改为平均值：总收益的平均值
        self.consumer_surplus = round(sum(c.total_revenue for c in consumer_list) / len(consumer_list), 4) if consumer_list else 0
        # 企业剩余改为平均值：利润的平均值  
        self.firm_surplus = round(sum(f.profit for f in firm_list) / len(firm_list), 4) if firm_list else 0
        self.temp_memory['total_consumer_surplus'] = self.consumer_surplus
        self.temp_memory['total_firm_surplus'] = self.firm_surplus

    def calculate_surplus_product_design(self, consumer_list, firm_list):
        self.consumer_surplus = 0.0
        self.firm_surplus = 0.0
        for c in consumer_list:
            self.consumer_surplus_product_design+=c.utility
        for f in firm_list:
            profit=0.0
            for c in consumer_list:
                if c.index==1:
                    q=f.personalized_products[c.index][1]
                    profit+=f.personalized_products[c.index][1]-q ** 2 / 2
                if c.index==2:
                    q=f.personalized_products[c.index][1]
                    profit+=f.public_menu[1]-q ** 2 / 2
            self.firm_surplus_product_design+=profit
        self.consumer_surplus_product_design/=len(consumer_list)
        self.firm_surplus_product_design/=len(firm_list)
        self.temp_memory['total_consumer_surplus'] = self.consumer_surplus_product_design
        self.temp_memory['total_firm_surplus'] = self.firm_surplus_product_design
    def update_memory(self, round_num=None, model_name=None):
        round_number = round_num + 1 if round_num is not None else 1  # 修复None问题
        current_round_state = {'round': round_number}
        keys_to_copy = ['num_consumers', 'num_firms', 'firm_list_prices', 'firm_personalized_prices',
                        'purchase_behavior', 'firm_sales', 'total_consumer_surplus', 'total_firm_surplus',
                        'actual_share_rate']
        for key in keys_to_copy:
            if key in self.temp_memory: 
                current_round_state[key] = copy.deepcopy(self.temp_memory[key])
        self.history_memory.append(current_round_state)
        if self.memory_distill: 
            self.update_memory_distill(round_num, model_name)

    def get_memory_for_prompt(self):
        return "Platform history not used."

    def update_memory_distill(self, round_num, model_name=None):
        pass

    @property
    def memory(self):
        return self.history_memory

    @memory.setter
    def memory(self, value):
        if isinstance(value, list):
            self.history_memory = value
        else:
            pass


class Broadcaster(AgentBase):
    def __init__(
            self,
            top_k_consumers: int = 3,
            top_n_firms: int = 3,
            distill_broadcast: bool = False,
            start_broadcast_round: int = 2,  # 从第几轮开始广播
            broadcast_history_window: int = 1,  # 考虑的历史窗口大小
            model_config_name="gpt-config",
            model_names=None,
            consumer_graph=None,  # <-- Add graph argument
            enable_cot=False
    ):
        super().__init__(name='broadcaster', model_config_name=model_config_name, use_memory=False)
        self.model_names = model_names
        self.model_config_name = model_config_name  # 用于获取蒸馏模型

        # 广播控制参数
        self.top_k_consumers = top_k_consumers
        self.top_n_firms = top_n_firms
        self.distill_broadcast = distill_broadcast
        self.start_broadcast_round = start_broadcast_round
        self.broadcast_history_window = broadcast_history_window
        self.enable_cot = enable_cot

        # 网络结构
        self.consumer_graph = consumer_graph
        self.network_type = "fully_connected" if consumer_graph is None else consumer_graph.graph.get('type',
                                                                                                      'unknown')  # Get type if set, else assume custom
        if consumer_graph is not None and self.network_type == 'unknown':  # Try inferring if not explicitly set
            if isinstance(consumer_graph, nx.Graph):
                # Simple check, might not be accurate for all generation methods
                if nx.is_isomorphic(consumer_graph, nx.complete_graph(len(consumer_graph.nodes))):
                    self.network_type = "fully_connected"
                    # Add more specific checks if needed, e.g., based on graph properties or generation method attributes
                else:
                    print(
                        f"Warning: Broadcaster received a graph but couldn't determine its standard type. Assuming custom network.")
                    self.network_type = "custom"  # or keep unknown

        # 存储历史轮次的临时记忆
        self.consumer_mems_history = []  # 存储过去 window 轮的消费者记忆
        self.firm_mems_history = []  # 存储过去 window 轮的公司记忆

        self.last_consumer_mems = []
        self.last_firm_mems = []

        # 为下一轮生成的广播消息 (消费者消息变为字典或单个全局消息)
        self.messages_for_decide_share_per_consumer = {}  # 个性化消息
        self.global_consumer_message = ""  # 全局消息

        # self.message_for_decide_search = "" # Search is not using broadcast currently
        self.message_for_decide_purchase = ""  # Assumed global or not network-dependent
        self.message_for_set_price = ""  # Assumed global
        self.message_for_set_price_pricing = ""  # Assumed global
        self.message_for_set_personalized_price = ""  # Assumed global

    def _get_distill_model(self):
        """获取用于蒸馏的模型实例 (需要 AgentBase 实例来获取模型)"""
        if self.model_names is None:  # 修复None问题
            return None
            
        distill_model_name = self.model_names.get('distill_broadcast', self.model_config_name)
        if distill_model_name:
            max_retries = 3
            retry_delay = 5

            for attempt in range(max_retries):
                try:
                    original_config = self.model_config_name
                    self.model_config_name = distill_model_name
                    temp_agent = AgentBase(name='temp_distiller', model_config_name=distill_model_name)
                    model = temp_agent.model
                    self.model_config_name = original_config
                    return model
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Broadcaster 获取蒸馏模型错误: {e}. 重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                    else:
                        print(f"Broadcaster 放弃获取蒸馏模型 {distill_model_name}")
                        return None
        return None

    # 从 Consumer 和 Firm 列表收集上一轮的临时记忆
    def get_last_round_mems(self, consumer_list: List[Consumer], firm_list: List[Firm]):
        """收集刚结束回合的 Agent 临时记忆，用于生成下一回合的广播消息，并保持历史窗口"""
        # 收集当前轮次的记忆
        current_consumer_mems = [copy.deepcopy(c.temp_memory) for c in consumer_list]
        current_firm_mems = [copy.deepcopy(f.temp_memory) for f in firm_list]

        self.last_consumer_mems = current_consumer_mems
        self.last_firm_mems = current_firm_mems

        self.consumer_mems_history.append(current_consumer_mems)
        self.firm_mems_history.append(current_firm_mems)

        if len(self.consumer_mems_history) > self.broadcast_history_window:
            self.consumer_mems_history = self.consumer_mems_history[-self.broadcast_history_window:]
        if len(self.firm_mems_history) > self.broadcast_history_window:
            self.firm_mems_history = self.firm_mems_history[-self.broadcast_history_window:]

        print("[DEBUG] Broadcaster collected last round memories:")
        if current_consumer_mems:
            print(
                f"  Sample Consumer Mem (idx {current_consumer_mems[0].get('index')}): "
                f"{ {k: v for k, v in current_consumer_mems[0].items() if k != 'searched_firms'} }"
            )
        if current_firm_mems:
            print(f"  Sample Firm Mem (idx {current_firm_mems[0].get('index')}): {current_firm_mems[0]}")

        print(f"  History window size: Consumer={len(self.consumer_mems_history)}, Firm={len(self.firm_mems_history)}")

    def _get_all_mems_in_window(self, is_consumer=True):
        """获取历史窗口内的所有记忆"""
        all_mems = []

        # 根据类型选择记忆历史
        history = self.consumer_mems_history if is_consumer else self.firm_mems_history

        # 合并历史窗口内的所有记忆
        for mems_in_round in history:
            all_mems.extend(mems_in_round)

        return all_mems

    def _distill_info(self, info_list: List[Dict], context: str) -> str:
        """使用 LLM 总结信息列表 (例如, top K 消费者记忆)"""
        if not self.distill_broadcast or not info_list:
            return str(info_list)  # 如果不蒸馏或列表为空，返回原始列表字符串

        base_content = f"Data to summarize:\n{str(info_list)}\n\nSummary:"
        if self.enable_cot:
            enhanced_content = f"{base_content}\n\nThink step by step:\n1. Identify key patterns in the data\n2. Analyze important trends and changes\n3. Extract the most valuable information\n4. Generate a concise and accurate summary\n\nPlease follow these steps and provide your final summary:"
        else:
            enhanced_content = base_content
            
        prompt = [
            {"role": "system", "content": f"请简洁地总结上一轮的以下关键信息，聚焦于 {context}。"},
            {"role": "user", "content": enhanced_content}
        ]

        try:
            model = self._get_distill_model()  # 获取蒸馏模型
            if model:
                summary = model(prompt).text
                return summary
            else:
                print("无法获取蒸馏模型，跳过蒸馏步骤。")
                return f"[蒸馏跳过]: {str(info_list)}"
        except Exception as e:
            print(f"广播信息蒸馏失败 ({context}): {e}")
            return f"[总结错误]: {str(info_list)}"  # 出错时返回错误信息和原始数据

    def _format_broadcast_message(self, mem_list: List[Dict], sort_key: str, top_n: int, name_prefix: str,
                                  context_desc: str) -> str:
        """辅助函数：对给定的记忆列表进行排序、选择 Top N 并格式化广播消息"""
        if not mem_list:
            return f"没有来自 {name_prefix} 的相关数据可广播 ({context_desc})。"

        # 按指定键排序 (降序), 仅包括有有效排序键的字典
        sorted_mems = sorted(
            [m for m in mem_list if isinstance(m.get(sort_key), (int, float))],
            key=lambda x: x.get(sort_key, -float('inf')),
            reverse=True
        )

        # 确定选择数量 (top_n == -1 表示全部)
        num_to_select = len(sorted_mems) if top_n == -1 else min(top_n, len(sorted_mems))

        if num_to_select <= 0:
            return f"没有有效的 {name_prefix} 数据可供选择 (基于 {sort_key}, {context_desc})。"

        top_mems = sorted_mems[:num_to_select]

        # 蒸馏或格式化
        header = f"来自 Top {num_to_select} {name_prefix} (按 {sort_key} 排序) 关于 {context_desc} 的信息:"
        if self.distill_broadcast:
            summary = self._distill_info(top_mems, context=f"{name_prefix} {context_desc}")
            return f"{header}\n{summary}"
        else:
            formatted_list = []
            for i, mem in enumerate(top_mems):
                idx = mem.get('index', f'排名 {i + 1}')
                sort_val = mem.get(sort_key, 'N/A')
                sort_val_str = f"{sort_val:.2f}" if isinstance(sort_val, (int, float)) else str(sort_val)

                # 不再排除理由等字段，仅排除 index 和 sort_key
                relevant_data = {k: v for k, v in mem.items() if k not in ['index', sort_key]}
                formatted_list.append(f" - {name_prefix}{idx} ({sort_key}:{sort_val_str}): {relevant_data}")
            return f"{header}\n" + "\n".join(formatted_list)

    # 修改: 生成所有决策步骤的广播消息
    def generate_all_messages(self, num_consumers: int):  # Add num_consumers
        """在回合开始时调用，使用历史窗口内收集的记忆生成广播消息"""
        # 重置消息
        self.messages_for_decide_share_per_consumer = {}
        self.global_consumer_message = ""
        self.message_for_decide_purchase = ""
        self.message_for_set_price = ""
        self.message_for_set_price_pricing = ""
        self.message_for_set_personalized_price = ""

        # 如果历史窗口为空，则设置空消息
        if not self.consumer_mems_history or not self.firm_mems_history:
            empty_msg = "--- 广播信息尚未生成 (无历史数据) ---"
            self.global_consumer_message = empty_msg
            # self.message_for_decide_search = empty_msg # Currently not used
            self.message_for_decide_purchase = empty_msg
            self.message_for_set_price = empty_msg
            self.message_for_set_price_pricing = empty_msg
            self.message_for_set_personalized_price = empty_msg
            # Ensure personalized dict is also empty/default
            for i in range(num_consumers):
                self.messages_for_decide_share_per_consumer[i] = empty_msg
            return

        # --- 生成公司广播信息 (保持全局) ---
        all_firm_mems_in_window = [m for round_mems in self.firm_mems_history for m in round_mems]
        share_rates = [m.get('actual_share_rate') for m in all_firm_mems_in_window if
                       isinstance(m.get('actual_share_rate'), float)]
        avg_share_rate = np.mean(share_rates) if share_rates else 0.5
        avg_share_rate_info = f"过去 {len(self.firm_mems_history)} 轮市场平均实际分享率: {avg_share_rate:.2f}"

        firm_context_desc = "定价与利润"
        firm_broadcast_info = self._format_broadcast_message(
            all_firm_mems_in_window, 'profit', self.top_n_firms, '公司', firm_context_desc
        )
        firm_broadcast_header = f"--- 全局市场历史信息广播 (窗口={len(self.firm_mems_history)}轮) ---"

        self.message_for_decide_purchase = f"{firm_broadcast_header}\n{firm_broadcast_info}\n-------------------------"  # Example for purchase decision
        self.message_for_set_price = f"{firm_broadcast_header}\n{avg_share_rate_info}\n{firm_broadcast_info}\n-------------------------"  # Example for set_price
        self.message_for_set_price_pricing = self.message_for_set_price  # Assuming same info needed
        self.message_for_set_personalized_price = self.message_for_set_price  # Assuming same info needed

        # --- 生成消费者广播信息 (根据网络类型) ---
        all_consumer_mems_in_window = [m for round_mems in self.consumer_mems_history for m in round_mems]
        consumer_context_desc = "分享决策与收益"
        consumer_broadcast_header = f"--- 邻居历史信息广播 (窗口={len(self.consumer_mems_history)}轮) ---"

        if self.network_type == "fully_connected" or self.consumer_graph is None:
            # 全局广播模式
            self.global_consumer_message = self._format_broadcast_message(
                all_consumer_mems_in_window, 'total_revenue', self.top_k_consumers, '消费者', consumer_context_desc
            )
            self.global_consumer_message = f"{consumer_broadcast_header.replace('邻居', '全局')}\n{self.global_consumer_message}\n-------------------------"
            # Make sure personalized dict has default message for consistency in main loop
            for i in range(num_consumers):
                self.messages_for_decide_share_per_consumer[i] = self.global_consumer_message
        else:
            # 基于网络的个性化广播
            if not isinstance(self.consumer_graph, nx.Graph):
                print("Error: Consumer graph is not a valid networkx graph. Cannot generate neighbor messages.")
                # Provide a default empty message
                for i in range(num_consumers):
                    self.messages_for_decide_share_per_consumer[i] = "--- 无法生成邻居广播信息 (图错误) ---"
                return

            self.global_consumer_message = ""  # Clear global message if using network
            for i in range(num_consumers):
                if i not in self.consumer_graph.nodes:
                    self.messages_for_decide_share_per_consumer[i] = "--- 无法生成邻居广播信息 (节点不在图中) ---"
                    continue

                neighbors_indices = list(self.consumer_graph.neighbors(i))
                # Filter memories to only include neighbors' memories from the history window
                neighbor_mems = [
                    mem for round_mems in self.consumer_mems_history
                    for mem in round_mems
                    if mem.get('index') in neighbors_indices
                ]

                personalized_message = self._format_broadcast_message(
                    neighbor_mems, 'total_revenue', self.top_k_consumers, '邻居消费者', consumer_context_desc
                )
                self.messages_for_decide_share_per_consumer[
                    i] = f"{consumer_broadcast_header}\n{personalized_message}\n-------------------------"

        # 添加: 打印生成的广播消息样本用于调试 (可选)
        # print("[DEBUG] Generated Broadcast Messages:")
        # if self.global_consumer_message: print(f"  Global Consumer Msg: {self.global_consumer_message[:200]}...")
        # if self.messages_for_decide_share_per_consumer: print(f"  Personalized C0 Msg: {self.messages_for_decide_share_per_consumer.get(0, '')[:200]}...")
        # print(f"  Set Price Msg: {self.message_for_set_price[:200]}...")
