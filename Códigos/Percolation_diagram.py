import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def create_grid_graph(L):
    G = nx.grid_2d_graph(L, L)
    return G

def percolate_grid(G, p):
    for edge in list(G.edges()):
        if np.random.rand() > p:
            G.remove_edge(*edge)
    return G

def find_percolating_cluster(G):
    top_nodes = [node for node in G.nodes() if node[1] == 0]
    bottom_nodes = [node for node in G.nodes() if node[1] == L-1]
    
    for source in top_nodes:
        for target in bottom_nodes:
            if nx.has_path(G, source, target):
                return nx.shortest_path(G, source=source, target=target)
    
    return []

def plot_percolation(G, percolating_path, title, ax):
    pos = {node: (node[0], L - 1 - node[1]) for node in G.nodes()}
    
    nx.draw(G, pos, node_size=10, node_color="black", with_labels=False, ax=ax)
    if percolating_path:
        edges = list(zip(percolating_path[:-1], percolating_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='blue', width=2.0, ax=ax)
    
    ax.set_title(title, fontsize=35)

L = 10 # Size of the grid
p_values = [0.25, 0.55, 0.9]  # Different probabilities
titles = ["$p<p_c$", "$p=p_c$", "$p>p_c$"]

fig, axes = plt.subplots(1, 3, figsize=(15, 7))

for i, p in enumerate(p_values):
    G = create_grid_graph(L)
    G = percolate_grid(G, p)
    percolating_path = find_percolating_cluster(G)
    title = titles[i]
    plot_percolation(G, percolating_path, title, ax=axes[i])

plt.tight_layout()
plt.show()
