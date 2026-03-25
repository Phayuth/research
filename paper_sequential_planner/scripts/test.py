import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


# ----------------------------
# Node sparsification (against existing graph)
# ----------------------------
def filter_new_points(points, new_points, eps):
    if len(points) == 0:
        return new_points

    tree = KDTree(points)
    keep = []
    for p in new_points:
        if len(tree.query_ball_point(p, eps)) == 0:
            keep.append(p)

    return np.array(keep) if len(keep) > 0 else np.empty((0, points.shape[1]))


# ----------------------------
# Local connection (kNN)
# ----------------------------
def connect_new_nodes(points, graph, start_idx, k):
    tree = KDTree(points)

    for i in range(start_idx, len(points)):
        dists, idx = tree.query(points[i], k + 1)

        for j, d in zip(idx[1:], dists[1:]):
            if j == i:
                continue

            graph[i].append((j, float(d)))
            graph[j].append((i, float(d)))  # undirected

    print(graph)


# ----------------------------
# Triangle-based edge pruning (local)
# ----------------------------
def prune_edges_local(graph, nodes, delta=0.1):
    for u in nodes:
        new_edges = []

        for v, d_uv in graph[u]:
            keep = True

            for w, d_uw in graph[u]:
                if w == v:
                    continue

                for x, d_wv in graph[w]:
                    if x == v and d_uw + d_wv <= (1 + delta) * d_uv:
                        keep = False
                        break
                if not keep:
                    break

            if keep:
                new_edges.append((v, d_uv))

        graph[u] = new_edges


# ----------------------------
# Incremental step
# ----------------------------
def incremental_step(points, graph, new_points, eps, k, delta):
    # 1. sparsify new samples
    new_points = filter_new_points(points, new_points, eps)
    if len(new_points) == 0:
        return points, graph

    # 2. append
    start_idx = len(points)
    points = np.vstack([points, new_points]) if len(points) else new_points

    # init graph entries
    for i in range(start_idx, len(points)):
        graph[i] = []

    # 3. connect
    connect_new_nodes(points, graph, start_idx, k)

    # 4. prune (only affected nodes)
    affected = list(range(start_idx, len(points)))
    prune_edges_local(graph, affected, delta)

    return points, graph


# ----------------------------
# Visualization
# ----------------------------
def plot_graph(points, graph, title=""):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=30)

    for i, neighbors in graph.items():
        for j, _ in neighbors:
            plt.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                "k-",
                alpha=0.3,
            )

    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.grid()
    plt.show()


# ----------------------------
# Main loop
# ----------------------------
def run_incremental_demo():
    np.random.seed(0)

    points = np.empty((0, 2))
    graph = {}

    eps = 0.05  # node sparsity
    k = 6  # connectivity
    delta = 0.1  # edge pruning strength

    for it in range(10):
        new_points = np.random.rand(20, 2)

        points, graph = incremental_step(points, graph, new_points, eps, k, delta)

        print(
            f"iter {it}: nodes={len(points)}, edges={sum(len(v) for v in graph.values())}"
        )

        if it % 3 == 0:
            plot_graph(points, graph, title=f"Iteration {it}")

    plot_graph(points, graph, title="Final roadmap")

    # new_points = np.random.rand(20, 2)
    # points, graph = incremental_step(points, graph, new_points, eps, k, delta)
    # print(f"edges={sum(len(v) for v in graph.values())}")
    # plot_graph(points, graph, title="Final roadmap")


if __name__ == "__main__":
    np.random.seed(20)

    run_incremental_demo()
    # points = np.random.rand(10, 2)
    # new_points = np.random.rand(20, 2)
    # eps = 0.05
    # k = 6
    # delta = 0.1

    # tree = KDTree(points)
    # pn = tree.query_ball_point(new_points, eps)
    # keepbool = np.array([len(neighbors) == 0 for neighbors in pn])
    # pkeeps = new_points[keepbool]

    # fig, ax = plt.subplots()
    # ax.scatter(points[:, 0], points[:, 1], s=50, label="Existing")
    # ax.scatter(new_points[:, 0], new_points[:, 1], s=50, label="New")
    # ax.scatter(
    #     pkeeps[:, 0],
    #     pkeeps[:, 1],
    #     s=100,
    #     facecolors="none",
    #     edgecolors="r",
    #     label="Kept",
    # )
    # ax.set_title("Before filtering")
    # ax.set_aspect("equal")
    # ax.legend()
    # plt.grid()
    # plt.show()
