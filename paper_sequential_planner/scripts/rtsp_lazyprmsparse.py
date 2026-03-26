import os
import numpy as np
import matplotlib.pyplot as plt
from paper_sequential_planner.scripts.geometric_ellipse import *
from paper_sequential_planner.experiments.env_planarrr import *
from scipy.spatial import KDTree
import heapq
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


robot = PlanarRR()
scene = RobotScene(robot, None)
cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))


def bulk_collisioncheck(X):
    print(X.shape)
    Xresult = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        q = X[i, :].reshape(-1, 1)
        best, res = scene.distance_to_obstacles(q)
        if best["distance"] <= 0:
            Xresult[i] = 1  # in collision
        else:
            Xresult[i] = 0  # free
    return Xresult


def is_node_in_collision(q):
    best, res = scene.distance_to_obstacles(q)
    return best["distance"] <= 0


def is_edge_in_collision(q1, q2):
    qs = np.linspace(q1, q2, num=10)
    for q in qs:
        if is_node_in_collision(q):
            return True
    return False


Q = np.array(
    [
        [-1.0, 2.5],
        [1.0, 2.5],
        [0.15, 0.60],
        [2.5, 1.5],
        [-2.5, -1.5],
        [2.40, -0.4],
        [-2.0, 2.5],
        [1.0, -2.0],
        [-3.0, 0.0],
        [-3.0, 2.5],
        [-1.4, 0.5],
        [-0.56, -2.37],
        [3.00, -2.00],
    ]
)
qs = Q[0].reshape(-1, 1)
qg = Q[6].reshape(-1, 1)
cmin = np.linalg.norm(qg - qs)
cMaxguess = 1.5 * cmin
la0 = cmin / 2
sa0 = 0
la1 = cMaxguess / 2
sa1 = cMaxguess / 2
la2 = cmin / 2
sa2 = cmin / 2
la3 = np.sqrt(cMaxguess**2 - cmin**2) / 2
sa3 = cMaxguess / 2
la4 = cmin / 2
sa4 = np.sqrt(cMaxguess**2 - cmin**2) / 2
la5 = np.sqrt(cMaxguess**2 - cmin**2) / 2
sa5 = cmin / 2


def build_graph(points, k, dist_threshold=np.inf):
    tree = KDTree(points)
    graph = {i: [] for i in range(len(points))}
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k + 1, distance_upper_bound=dist_threshold)
        for j, d in zip(idx[1:], dists[1:]):
            graph[i].append((j, float(d)))
    return graph, tree


def sparsify_nodes(points, eps):
    tree = KDTree(points)
    keep = np.ones(len(points), dtype=bool)

    for i, p in enumerate(points):
        if not keep[i]:
            continue
        idx = tree.query_ball_point(p, eps)
        for j in idx:
            if j != i:
                keep[j] = False

    return points[keep]


def prune_edges_triangle(graph, points, delta=0.1):
    new_graph = {i: [] for i in graph}

    for u in graph:
        for v, d_uv in graph[u]:
            keep = True

            for w, d_uw in graph[u]:
                if w == v:
                    continue

                # check if w connects to v
                for x, d_wv in graph[w]:
                    if x == v:
                        if d_uw + d_wv <= (1 + delta) * d_uv:
                            keep = False
                        break

                if not keep:
                    break

            if keep:
                new_graph[u].append((v, d_uv))

    return new_graph


def graph_mid_point_collision_prune(graph, points):
    new_graph = {i: [] for i in graph}
    for u in graph:
        for v, d_uv in graph[u]:
            # d = np.linalg.norm(points[u] - points[v])
            mid_point = (points[u] + points[v]) / 2
            if not is_node_in_collision(mid_point):
                new_graph[u].append((v, d_uv))
    return new_graph


def dijkstra_all_paths(graph, root):
    dist = {node: float("inf") for node in graph}
    parent = {node: None for node in graph}

    dist[root] = 0
    pq = [(0, root)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:
            new_dist = current_dist + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, parent


def dijkstra_bulk_paths(graph, root, targets):
    dist = {node: float("inf") for node in graph}
    parent = {node: None for node in graph}

    dist[root] = 0
    pq = [(0, root)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        if u in targets:
            targets.remove(u)
            if not targets:
                break

        for v, weight in graph[u]:
            new_dist = current_dist + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, parent


def reconstruct_path(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def get_path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        cost += np.linalg.norm(path[i + 1] - path[i])
    return cost


def allnode_RGG_sparse_robot():
    X = np.random.uniform(-np.pi, np.pi, size=(500, 2))
    Xs = bulk_collisioncheck(X)
    points = X[Xs == 0]  # only keep collision-free nodes
    points_sparse = sparsify_nodes(points, eps=0.05 * 2 * np.pi)  # node sparse
    print(f"==>> points.shape: \n{points.shape}")
    print(f"==>> points_sparse.shape: \n{points_sparse.shape}")

    k = 5
    graph, kdt = build_graph(points, k)
    graph_sparse, kdt_sparse = build_graph(points_sparse, k)
    graph_sparse = prune_edges_triangle(graph_sparse, points_sparse, delta=0.1)
    graph_sparse = graph_mid_point_collision_prune(graph_sparse, points_sparse)

    root = 0
    target = 82
    _, parent = dijkstra_all_paths(graph_sparse, root)
    pathid = reconstruct_path(parent, target)
    pathq_mid = points_sparse[pathid]
    pathq_full = np.vstack([points_sparse[root], pathq_mid, points_sparse[target]])
    print(f"==>> pathq_full: \n{pathq_full}")
    cost = get_path_cost(pathq_full)

    targets = [17, 124, 20, 57]
    _, parent_bulk = dijkstra_bulk_paths(graph_sparse, root, targets.copy())
    pathq_full_bulk = []
    cost_bulk = []
    for t in targets:
        pathid_bulk = reconstruct_path(parent_bulk, t)
        pathq_mid_bulk = points_sparse[pathid_bulk]
        pathq_full_bulk_i = np.vstack(
            [points_sparse[root], pathq_mid_bulk, points_sparse[t]]
        )
        cost_bulk_i = get_path_cost(pathq_full_bulk_i)
        print(f"==>> pathq_full_bulk_i for target {t}: \n{pathq_full_bulk_i}")
        print(f"==>> Cost to target {t}: {cost_bulk_i:.2f}")
        pathq_full_bulk.append(pathq_full_bulk_i)
        cost_bulk.append(cost_bulk_i)

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=100,
        c="lightgray",
        marker="x",
        label="Original Nodes",
    )
    ax.scatter(
        points_sparse[:, 0], points_sparse[:, 1], s=50, c="b", label="Sparse Nodes"
    )
    for i, node in enumerate(points_sparse):
        ax.text(node[0], node[1], str(i), fontsize=8, color="red", ha="right")

    for i, neighbors in graph.items():
        for j, _ in neighbors:
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                "r--",
                alpha=0.1,
            )

    for i, neighbors in graph_sparse.items():
        for j, _ in neighbors:
            ax.plot(
                [points_sparse[i, 0], points_sparse[j, 0]],
                [points_sparse[i, 1], points_sparse[j, 1]],
                "k-",
                alpha=0.4,
            )
    # ax.plot(
    #     pathq_full[:, 0],
    #     pathq_full[:, 1],
    #     "g-",
    #     linewidth=2,
    # )
    for p in pathq_full_bulk:
        ax.plot(
            p[:, 0],
            p[:, 1],
            "m*--",
            linewidth=5,
            markersize=10,
        )
    ax.scatter(
        points_sparse[root][0],
        points_sparse[root][1],
        s=100,
        c="r",
        marker="s",
        facecolors="none",
        label="Root Node",
    )
    ax.scatter(
        points_sparse[target][0],
        points_sparse[target][1],
        s=100,
        c="g",
        marker="s",
        facecolors="none",
        label="Target Node",
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # allnode_RGG_sparse()
    allnode_RGG_sparse_robot()
    # allnode_RGG_sparse_3d()
