import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def draw_weighted_graph(
    node_names,
    importance,
    adjacency,
    edge_threshold=0.15,  # 边的显示阈值
    node_size_scale=4000,  # 节点大小缩放
    edge_width_scale=6,    # 边粗细
    edge_label_fontsize=12, # 边权重字体大小
    label_fontsize=17,      # 节点标签字体大小
    layout_k=20,           # Spring 力导布局参数
    save_path="graph_1.png"
):
    # 解决你之前"图不刷新"的问题
    plt.clf()
    plt.close()

    G = nx.Graph()

    # -------------------------
    # 节点大小 / importance 越小越大
    # -------------------------
    importance = np.array(importance)
    inv_imp = 1 / importance
    node_sizes = node_size_scale * inv_imp / inv_imp.max()

    # -------------------------
    # 三组颜色 + 图例
    # -------------------------
    benefits = {"Search Cost Savings", "Positive Profit"}
    drawbacks = {"Privacy Loss", "Negative Profit", "Search Continuation Cost"}
    tradeoffs = {"Net Benefit", "Profit Maximization", "Loss Minimization"}

    color_map = []
    for nm in node_names:
        if nm in benefits:
            color_map.append("#66c2a5")  # green
        elif nm in drawbacks:
            color_map.append("#fc8d62")  # orange
        elif nm in tradeoffs:
            color_map.append("#8da0cb")  # blue
        else:
            color_map.append("#bdbdbd")  # fallback gray

    # -------------------------
    # 简写标签函数
    # -------------------------
    def short_label(name):
        if name == "Profit Maximization":
            return "PMax"
        if name == "Loss Minimization":
            return "LMin"
        return "".join([w[0] for w in name.split()])

    short_labels = [short_label(nm) for nm in node_names]

    # -------------------------
    # 建立节点
    # -------------------------
    for i, name in enumerate(node_names):
        G.add_node(i, label=short_labels[i], size=node_sizes[i], color=color_map[i])

    # -------------------------
    # 加边
    # -------------------------
    A = np.array(adjacency)
    n = len(node_names)
    for i in range(n):
        for j in range(i+1, n):
            w = A[i, j]
            if w >= edge_threshold:
                G.add_edge(i, j, weight=w)

    # 输出 Negative Profit 和 Positive Profit 的边权重
    id_pos = node_names.index("Positive Profit")
    id_neg = node_names.index("Negative Profit")
    print(f"Edge weight between Positive Profit and Negative Profit: {A[id_pos, id_neg]:.4f}")

    # -------------------------
    # spring layout 优化布局
    # -------------------------
    pos = nx.spring_layout(G, seed=42, k=layout_k, iterations=200)

    # -------------------------
    # 绘制节点
    # -------------------------
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[G.nodes[n]['size'] for n in G.nodes],
        node_color=[G.nodes[n]['color'] for n in G.nodes],
        edgecolors="black",
        linewidths=1.2
    )

    # -------------------------
    # 绘制边（权重→粗细）
    # -------------------------
    edges = G.edges()
    widths = [edge_width_scale * G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.8)

    # -------------------------
    # 边权重标签（排除 Positive Profit 和 Negative Profit 的边）
    # -------------------------
    edge_labels = {}
    for u, v in edges:
        if {u, v} == {id_pos, id_neg}:
            continue
        edge_labels[(u, v)] = f"{G[u][v]['weight']:.2f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=edge_label_fontsize)

    # -------------------------
    # 节点名字放在圆圈里：使用缩写标签
    # -------------------------
    labels = {i: G.nodes[i]['label'] for i in G.nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=label_fontsize,  # 字体较小可放进圈内
        font_color="black",
        verticalalignment="center",
        horizontalalignment="center"
    )

    # ==================================================
    # ★ 修改：去掉左上角图例，注释掉下面的图例代码
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

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图已保存至：{save_path}")

    plt.show()

# ---------------------------------------------------
# 直接运行下面即可画图
# ---------------------------------------------------
node_names = [
    "Search Cost Savings", "Positive Profit", "Privacy Loss", "Negative Profit",
    "Search Continuation Cost", "Net Benefit", "Profit Maximization", "Loss Minimization"
]

importance = [1.316934720908231, 1.153263954588458, 1.9366130558183539, 2.8297067171239356,
              2.492904446546831, 1.89120151371807, 2.1324503311258276, 2.7332071901608326]

adjacency = [[0.0, 0.8694418164616841, 0.9583727530747398, 0.0, 0.3784295175023652, 0.9631031220435194, 0.47871333964049195, 0.006622516556291391],
             [0.8694418164616841, 0.0, 0.6944181646168401, 0.15042573320719016, 0.03216650898770104, 0.6083254493850521, 0.7152317880794702, 0.12298959318826869],
             [0.9583727530747398, 0.6944181646168401, 0.0, 0.05771050141911069, 0.000946073793755913, 0.9035004730368968, 0.09366130558183539, 0.17691579943235572],
             [0.0, 0.15042573320719016, 0.05771050141911069, 0.0, 0.039735099337748346, 0.02270577105014191, 0.01229895931882687, 0.10690633869441817],
             [0.3784295175023652, 0.03216650898770104, 0.000946073793755913, 0.039735099337748346, 0.0, 0.12866603595080417, 0.1750236518448439, 0.013245033112582781],
             [0.9631031220435194, 0.6083254493850521, 0.9035004730368968, 0.02270577105014191, 0.12866603595080417, 0.0, 0.45506149479659413, 0.12771996215704826],
             [0.47871333964049195, 0.7152317880794702, 0.09366130558183539, 0.01229895931882687, 0.1750236518448439, 0.45506149479659413, 0.0, 0.05203405865657521],
             [0.006622516556291391, 0.12298959318826869, 0.17691579943235572, 0.10690633869441817, 0.013245033112582781, 0.12771996215704826, 0.05203405865657521, 0.0]]

draw_weighted_graph(node_names, importance, adjacency)