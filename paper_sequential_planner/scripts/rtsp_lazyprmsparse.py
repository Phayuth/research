import os
import numpy as np
import matplotlib.pyplot as plt
from paper_sequential_planner.scripts.geometric_ellipse import *
from paper_sequential_planner.experiments.env_planarrr import *
from scipy.spatial import KDTree
import heapq
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, Delaunay
from sklearn.metrics.pairwise import euclidean_distances
import trimesh

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


def allnode_RGG_sparse():
    points = np.random.uniform(-np.pi, np.pi, size=(200, 2))
    points_sparse = sparsify_nodes(points, eps=0.05 * 2 * np.pi)  # node sparse

    k = 5
    graph, kdt = build_graph(points, k)
    graph_sparse, kdt_sparse = build_graph(points_sparse, k)
    graph_sparse = prune_edges_triangle(graph_sparse, points_sparse, delta=0.1)

    rootnode = points_sparse[0]
    print(f"==>> rootnode: \n{rootnode}")
    goalnode = points_sparse[1]
    print(f"==>> goalnode: \n{goalnode}")
    gg, ggi = kdt_sparse.query(goalnode, k=10)
    ggiq = points_sparse[ggi]
    ch = ConvexHull(ggiq)
    chvq = ggiq[ch.vertices]

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
    ax.scatter(
        rootnode[0], rootnode[1], s=100, c="r", marker="s", label="Root Node"
    )
    ax.scatter(
        goalnode[0], goalnode[1], s=100, c="g", marker="^", label="Goal Node"
    )

    cluster_polygon = plt.Polygon(
        chvq,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        label="Convex Hull of Neighbors",
    )
    ax.add_patch(cluster_polygon)

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

    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()


def allnode_RGG_sparse_3d():
    # points = np.random.rand(100, 3)
    points = np.random.uniform(-np.pi, np.pi, size=(200, 3))

    # --- 1. node sparsification ---
    points = sparsify_nodes(points, eps=0.05 * 2 * np.pi)

    k = 5
    graph, kdt = build_graph(points, k)

    # --- 2. edge pruning ---
    graph = prune_edges_triangle(graph, points, delta=0.1)

    rootid = 0
    goalid = 1
    rootnode = points[rootid]
    goalnode = points[goalid]

    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=np.pi)
    box = trimesh.creation.box(extents=(2 * np.pi, 2 * np.pi, 2 * np.pi))
    box.visual.face_colors = [100, 150, 255, 40]
    scene.add_geometry(box)
    scene.add_geometry(axis)

    for i, neighbors in graph.items():
        for j, _ in neighbors:
            line = trimesh.load_path(
                np.array([points[i], points[j]]), color=[0, 0, 0, 40]
            )
            scene.add_geometry(line)
    scene.show()


def graph_mid_point_collision_prune(graph, points):
    new_graph = {i: [] for i in graph}
    for u in graph:
        for v, d_uv in graph[u]:
            # d = np.linalg.norm(points[u] - points[v])
            mid_point = (points[u] + points[v]) / 2
            if not is_node_in_collision(mid_point):
                new_graph[u].append((v, d_uv))
    return new_graph


def allnode_RGG_sparse_robot():
    X = np.random.uniform(-np.pi, np.pi, size=(500, 2))
    print(f"==>> X.shape: \n{X.shape}")
    Xs = bulk_collisioncheck(X)
    points = X[Xs == 0]  # only keep collision-free nodes
    points_sparse = sparsify_nodes(points, eps=0.05 * 2 * np.pi)  # node sparse
    print(f"==>> points_sparse.shape: \n{points_sparse.shape}")

    k = 5
    graph, kdt = build_graph(points, k)
    graph_sparse, kdt_sparse = build_graph(points_sparse, k)
    graph_sparse = prune_edges_triangle(graph_sparse, points_sparse, delta=0.1)
    graph_sparse = graph_mid_point_collision_prune(graph_sparse, points_sparse)

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

    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()


if __name__ == "__main__":
    allnode_RGG_sparse()
    allnode_RGG_sparse_robot()
    allnode_RGG_sparse_3d()
