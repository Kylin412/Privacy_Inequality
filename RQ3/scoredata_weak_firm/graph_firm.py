import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_weighted_graph(
        node_names,
        importance,
        adjacency,
        edge_threshold=0.1,
        node_size_scale=3500,
        edge_width_scale=4,
        edge_label_fontsize=9,
        layout_k=12,
        label_fontsize=18,
        save_path="graph_gpt-firm.png"
):
    plt.clf()
    plt.close()

    G = nx.Graph()

    # 节点大小计算（不变）
    importance = np.array(importance)
    inv_imp = 1 / importance
    node_sizes = node_size_scale * inv_imp / inv_imp.max()

    # 颜色分类（不变）
    benefits = {"Profit Margin", "Customer Loyalty"}
    drawbacks = {"Competitive Pressure", "Customer Churn Risk", "Demand Uncertainty"}
    tradeoffs = {"Net Benefit", "Profit Maximization", "Loss Minimization"}

    color_map = []
    for nm in node_names:
        if nm in benefits:
            color_map.append("#66c2a5")
        elif nm in drawbacks:
            color_map.append("#fc8d62")
        elif nm in tradeoffs:
            color_map.append("#8da0cb")
        else:
            color_map.append("#bdbdbd")

    # 简写函数（不变）
    def short_label(name):
        if name == "Profit Maximization":
            return "PMax"
        if name == "Loss Minimization":
            return "LMin"
        return "".join([w[0] for w in name.split()])

    short_labels = [short_label(nm) for nm in node_names]

    # 添加节点（不变）
    for i, name in enumerate(node_names):
        G.add_node(i,
                   label=short_labels[i],
                   full=name,
                   size=node_sizes[i],
                   color=color_map[i])

    # 添加边（不变）
    A = np.array(adjacency)
    n = len(node_names)

    for i in range(n):
        for j in range(i + 1, n):
            w = A[i, j]
            if w >= edge_threshold:
                G.add_edge(i, j, weight=w)

    # 布局（不变）
    pos = nx.spring_layout(G, seed=42, k=layout_k, iterations=200)

    # 节点绘制（不变）
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[G.nodes[n]["size"] for n in G.nodes],
        node_color=[G.nodes[n]["color"] for n in G.nodes],
        edgecolors="black",
        linewidths=1.2
    )

    # 边绘制（不变）
    edges = G.edges()
    widths = [edge_width_scale * G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.8)

    # ==================================================
    # ★ 关键修改区域：只隐藏 PMax-LMin 的权重标签
    # ==================================================
    id_pmax = node_names.index("Profit Maximization")
    id_lmin = node_names.index("Loss Minimization")

    edge_labels = {}
    is_pmax_lmin_labeled = False

    for u, v in edges:
        # 检查当前边是否是 PMax 和 LMin 之间的连线
        if {u, v} == {id_pmax, id_lmin}:
            # 找到目标边，但跳过添加标签，从而隐藏其权重数字
            is_pmax_lmin_labeled = True  # 标记这条边存在，但未被标记
            continue

            # 为其他边添加权重标签
        edge_labels[(u, v)] = f"{G[u][v]['weight']:.2f}"

    # --------------------------
    # 调试信息：确认是否隐藏成功
    # --------------------------
    pmax_name = node_names[id_pmax]
    lmin_name = node_names[id_lmin]

    # 检查目标边是否存在于 edge_labels 字典中（它不应该存在）
    target_edge = tuple(sorted((id_pmax, id_lmin)))

    if target_edge in edge_labels or (id_pmax, id_lmin) in edge_labels or (id_lmin, id_pmax) in edge_labels:
        print(f"【错误】目标边 ({pmax_name}, {lmin_name}) 的权重仍然存在于标签字典中！")
    else:
        print(f"【成功】目标边 ({pmax_name}, {lmin_name}) 的权重标签已成功从图中移除。")

    if not is_pmax_lmin_labeled:
        print(
            f"【注意】PMax-LMin 的边权重 {A[id_pmax, id_lmin]:.4f} 可能小于 edge_threshold={edge_threshold}，因此这条连线本身可能不存在。")

    # 边权重标签（绘制）
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=edge_label_fontsize
    )

    # 节点标签（不变）
    labels = {i: G.nodes[i]["label"] for i in G.nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=label_fontsize,
        font_color="black",
        verticalalignment="center",
        horizontalalignment="center"
    )

    # ==================================================
    # ★ 修改：注释掉图例部分
    # ==================================================
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor="#66c2a5", edgecolor="black", label="Benefits"),
    #     Patch(facecolor="#fc8d62", edgecolor="black", label="Drawbacks"),
    #     Patch(facecolor="#8da0cb", edgecolor="black", label="Trade-offs"),
    # ]
    # plt.legend(handles=legend_elements, loc="upper left", fontsize=10, frameon=True)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图已保存至：{save_path}")

    plt.show()


# ---------------------------------------------------
# 运行绘图示例
# ---------------------------------------------------

node_names = [
    "Profit Margin", "Customer Loyalty", "Competitive Pressure", "Customer Churn Risk",
    "Demand Uncertainty", "Net Benefit", "Profit Maximization", "Loss Minimization"
]

importance = [1.7103064066852367, 2.6908077994428967, 2.247910863509749, 2.796657381615599, 2.8050139275766015, 1.4707520891364902, 1.3008356545961002, 2.9080779944289694]

adjacency = [[0.0, 0.05013927576601671, 0.4986072423398329, 0.15877437325905291, 0.04178272980501393, 1.0, 0.947075208913649, 0.04735376044568245], [0.05013927576601671, 0.0, 0.0, 0.005571030640668524, 0.0, 0.011142061281337047, 0.052924791086350974, 0.002785515320334262], [0.4986072423398329, 0.0, 0.0, 0.08913649025069638, 0.013927576601671309, 0.22841225626740946, 0.2924791086350975, 0.0584958217270195], [0.15877437325905291, 0.005571030640668524, 0.08913649025069638, 0.0, 0.027855153203342618, 0.10584958217270195, 0.002785515320334262, 0.13649025069637882], [0.04178272980501393, 0.0, 0.013927576601671309, 0.027855153203342618, 0.0, 0.06406685236768803, 0.04456824512534819, 0.036211699164345405], [0.9972144846796658, 0.011142061281337047, 0.22841225626740946, 0.10863509749303621, 0.06406685236768803, 0.0, 0.9025069637883009, 0.10306406685236769], [0.947075208913649, 0.052924791086350974, 0.2924791086350975, 0.002785515320334262, 0.04456824512534819, 0.8997214484679665, 0.0, 0.022284122562674095], [0.04735376044568245, 0.002785515320334262, 0.0584958217270195, 0.1392757660167131, 0.036211699164345405, 0.10306406685236769, 0.033426183844011144, 0.0]]

draw_weighted_graph(
    node_names, importance, adjacency,
    edge_width_scale=6,  # 控制边的粗细
    edge_label_fontsize=12  # 控制边权重字体大小
)