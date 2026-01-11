import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_weighted_graph(
        node_names,
        importance,
        adjacency,
        edge_threshold=0.15,
        node_size_scale=3500,
        edge_width_scale=4,
        edge_label_fontsize=9,
        layout_k=12,
        label_fontsize=17,
        save_path="graph_grok-firm.png"
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
    id_pmax = node_names.index("Profit Margin")
    id_lmin = node_names.index("Loss Minimization")

    edge_labels = {}

    for u, v in edges:
        # 检查当前边是否是 PMax 和 LMin 之间的连线
        if {u, v} == {id_pmax, id_lmin}:
            # 找到目标边，但跳过添加标签，从而隐藏其权重数字
            continue

        # 为其他边添加权重标签
        edge_labels[(u, v)] = f"{G[u][v]['weight']:.2f}"

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

importance = [1.645083932853717, 2.2494004796163067, 1.8920863309352518, 2.7817745803357314,
              2.9568345323741005, 1.8489208633093526, 1.6810551558752997, 2.54916067146283]

adjacency = [
    [0.0, 0.36930455635491605, 0.60431654676259, 0.08393285371702638, 0.009592326139088728,
     0.9424460431654677, 0.7074340527577938, 0.24700239808153476],
    [0.36930455635491605, 0.0, 0.38609112709832133, 0.0, 0.0,
     0.15347721822541965, 0.27577937649880097, 0.0],
    [0.60431654676259, 0.38609112709832133, 0.0, 0.3261390887290168, 0.007194244604316547,
     0.5827338129496403, 0.4724220623501199, 0.09592326139088729],
    [0.08393285371702638, 0.0, 0.3261390887290168, 0.0, 0.007194244604316547,
     0.15347721822541965, 0.08393285371702638, 0.12470023980815348],
    [0.009592326139088728, 0.0, 0.007194244604316547, 0.007194244604316547, 0.0,
     0.03117505995203837, 0.019184652278177457, 0.011990407673860911],
    [0.9424460431654677, 0.15347721822541965, 0.5827338129496403, 0.15347721822541965,
     0.03117505995203837, 0.0, 0.7266187050359713, 0.2637889688249401],
    [0.7074340527577938, 0.27577937649880097, 0.4724220623501199, 0.08393285371702638,
     0.019184652278177457, 0.7266187050359713, 0.0, 0.15827338129496402],  # PMax-LMin = 0.158273... > 0.15
    [0.24700239808153476, 0.0, 0.09592326139088729, 0.12470023980815348,
     0.011990407673860911, 0.2637889688249401, 0.15827338129496402, 0.0]
]

draw_weighted_graph(
    node_names, importance, adjacency,
    edge_width_scale=6,  # 控制边的粗细
    edge_label_fontsize=12  # 控制边权重字体大小
)