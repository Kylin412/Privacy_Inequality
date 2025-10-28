import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 正确的节点名称
labels = [
    "F1 Profit Margin",
    "F2 Customer Loyalty",
    "F3 Competitive Pressure",
    "F4 Customer Churn Risk",
    "F5 Demand Uncertainty",
    "F6 Net Profit",
    "F7 Profit Maximization",
    "F8 Loss Minimization"
]

# 强竞争数据
vec_strong = [1.5, 2.85, 1.78, 2.82, 2.04, 1.05, 1.86, 2.58]
mat_strong = np.array([
    [0.0, 0.07, 0.0, 0.0, 0.0, 0.53, 0.11, 0.11],
    [0.07, 0.0, 0.01, 0.0, 0.0, 0.07, 0.01, 0.0],
    [0.0, 0.01, 0.0, 0.05, 0.31, 0.15, 0.0, 0.01],
    [0.0, 0.0, 0.03, 0.0, 0.13, 0.09, 0.0, 0.04],
    [0.0, 0.0, 0.31, 0.13, 0.0, 0.72, 0.0, 0.09],
    [0.53, 0.07, 0.15, 0.09, 0.68, 0.0, 0.16, 0.28],
    [0.11, 0.01, 0.0, 0.0, 0.05, 0.06, 0.0, 0.16],
    [0.11, 0.0, 0.01, 0.04, 0.09, 0.28, 0.16, 0.0]
])

# 弱竞争数据
vec_weak = [1.55, 2.83, 2.01, 2.68, 1.97, 1.03, 1.95, 2.76]
mat_weak = np.array([
    [0.0, 0.05, 0.0, 0.0, 0.0, 0.47, 0.01, 0.01],
    [0.05, 0.0, 0.01, 0.01, 0.0, 0.09, 0.0, 0.0],
    [0.0, 0.01, 0.0, 0.03, 0.27, 0.12, 0.01, 0.02],
    [0.0, 0.01, 0.01, 0.0, 0.26, 0.11, 0.0, 0.01],
    [0.0, 0.0, 0.27, 0.25, 0.0, 0.7, 0.01, 0.04],
    [0.47, 0.09, 0.12, 0.11, 0.67, 0.0, 0.06, 0.15],
    [0.01, 0.0, 0.01, 0.0, 0.05, 0.0, 0.0, 0.07],
    [0.01, 0.0, 0.02, 0.01, 0.04, 0.15, 0.07, 0.0]
])

def draw_network(mat, vec, labels, title, threshold=0.05):
    G = nx.Graph()
    n = len(vec)
    for i in range(n):
        G.add_node(i, size=(3.5 - vec[i]) * 400, label=labels[i])
    for i in range(n):
        for j in range(n):
            if mat[i, j] > threshold:
                G.add_edge(i, j, weight=mat[i, j])

    pos = nx.spring_layout(G, seed=42)
    edge_widths = [G[u][v]['weight'] * 8 for u, v in G.edges()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', edgecolors='gray')
    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in G.nodes()}, font_size=10)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.8)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

# 绘制强、弱竞争网络
draw_network(mat_strong, vec_strong, labels, "Strong Competition Network")
draw_network(mat_weak, vec_weak, labels, "Weak Competition Network")
