import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def robotsp_dijkstra_solve(qinit, Qorder):
    Layer = [qinit] + Qorder + [qinit]
    G = nx.DiGraph()
    positions = {}
    node_id = 0
    layers = []
    for lay in Layer:
        lid = []
        if lay.shape == (6,):
            positions[node_id] = lay
            G.add_node(node_id)
            lid.append(node_id)
            node_id += 1
        else:
            for l in lay:
                positions[node_id] = l
                G.add_node(node_id)
                lid.append(node_id)
                node_id += 1
        layers.append(lid)

    for i in range(len(layers) - 1):
        for src in layers[i]:
            for dst in layers[i + 1]:
                pos_src = positions[src]
                pos_dst = positions[dst]
                w = np.linalg.norm(pos_src - pos_dst)
                G.add_edge(src, dst, weight=w)

    config_path_id = nx.dijkstra_path(G, 0, node_id - 1)

    optimal_config = []
    for id in config_path_id:
        optimal_config.append(positions[id])

    return optimal_config


def robotsp_dijkstra_test(tasknumber=245, nodeeach=8):
    G = nx.DiGraph()
    layer_sizes = [1] + [nodeeach] * tasknumber + [1]
    positions = {}
    node_id = 0
    layers = []

    # Assign positions and create nodes
    for layer_idx, size in enumerate(layer_sizes):
        layer_nodes = []
        for i in range(size):
            # Spread nodes horizontally, layers vertically
            pos = np.array([i - (size - 1) / 2, layer_idx])
            G.add_node(node_id, pos=pos)
            positions[node_id] = pos
            layer_nodes.append(node_id)
            node_id += 1
        layers.append(layer_nodes)

    # Fully connect each layer to the next with Euclidean distance as weight
    for l in range(len(layers) - 1):
        for src in layers[l]:
            for dst in layers[l + 1]:
                pos_src = positions[src]
                pos_dst = positions[dst]
                weight = np.linalg.norm(pos_src - pos_dst)
                G.add_edge(src, dst, weight=weight)

    print("Nodes:", G.nodes(data=True))
    print("Edges:", [(u, v, d["weight"]) for u, v, d in G.edges(data=True)])

    path = nx.dijkstra_path(G, 0, nodeeach * tasknumber + 1)
    print("Shortest path is ", path)

    plt.figure(figsize=(10, 6))
    pos = {node: data["pos"] for node, data in G.nodes(data=True)}
    nx.draw(
        G, pos, with_labels=True, node_size=300, node_color="skyblue", arrows=True
    )
    plt.title("Neural Network Layer Graph")
    plt.xlabel("Horizontal Position")
    plt.ylabel("Layer (Vertical Position)")
    plt.show()


if __name__ == "__main__":
    robotsp_dijkstra_test()
