import os
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from problem_planarrr import PlanarRR, RobotScene
from geometric_ellipse import *
from scipy.spatial import KDTree
import heapq
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

shapes = {
    # "shape1": {"x": -0.7, "y": 1.3, "h": 2, "w": 2.2},
    "shape1": {"x": -0.7, "y": 2.1, "h": 2, "w": 2.2},
    "shape2": {"x": 2, "y": -2.0, "h": 1, "w": 4.0},
    "shape3": {"x": -3, "y": -3, "h": 1.25, "w": 2},
}
obstacles = [
    box(k["x"], k["y"], k["x"] + k["w"], k["y"] + k["h"]) for k in shapes.values()
]
robot = PlanarRR()
scene = RobotScene(robot, obstacles)


def collision_check(q):
    best, res = scene.distance_to_obstacles(q)
    if best["distance"] <= 0:
        return True
    else:
        return False


dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
datasetfree = dataset[dataset[:, 2] == -1]
datasetcoll = dataset[dataset[:, 2] == 1]


def build_graph(points, k, dist_thres=np.inf):
    tree = KDTree(points)
    graph = {i: [] for i in range(len(points))}
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k + 1, distance_upper_bound=dist_thres)
        for j, d in zip(idx[1:], dists[1:]):
            if j == len(points) or np.isinf(d):
                continue
            graph[i].append((int(j), float(d)))
    return graph, tree


# global space
lmts = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
Qful_snum = 1000
QfulRnd = np.random.uniform(
    lmts[:, 0], lmts[:, 1], size=(Qful_snum, lmts.shape[0])
)
print("Random configuration:", QfulRnd.shape)

# separate free and coll samples
QfulRndcheck = np.zeros((Qful_snum, 1))
for i in range(Qful_snum):
    q = QfulRnd[i, :].reshape(-1, 1)
    in_collision = collision_check(q)
    if in_collision:
        QfulRndcheck[i, 0] = 1
    else:
        QfulRndcheck[i, 0] = 0
QfulRndfree = QfulRnd[QfulRndcheck.flatten() == 0]
QfulRndcoll = QfulRnd[QfulRndcheck.flatten() == 1]
print(f"QfulRndfree: {QfulRndfree.shape}, QfulRndcoll: {QfulRndcoll.shape}")

# build graphs
gQfree_no_dist, tQfree_no_dist = build_graph(QfulRndfree, k=10)
knnradius_og = 0.5
knnradius_recal = rewire_radius(knnradius_og, 2, lmts, Qful_snum, rwfact=1.1)
print(f"Original kNN radius: {knnradius_og}")
print(f"Recalibrated kNN radius: {knnradius_recal:.2f}")
gQfree_dist, tQfree_dist = build_graph(
    QfulRndfree, k=10, dist_thres=knnradius_recal
)


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(datasetcoll[:, 0], datasetcoll[:, 1], c="brown", s=5)
ax.scatter(QfulRndfree[:, 0], QfulRndfree[:, 1], c="g", s=25, label="Qf_free")
ax.scatter(QfulRndcoll[:, 0], QfulRndcoll[:, 1], c="k", s=25, label="Qf_coll")
for i, g in gQfree_no_dist.items():
    for j, _ in g:
        ax.plot(
            [QfulRndfree[i, 0], QfulRndfree[j, 0]],
            [QfulRndfree[i, 1], QfulRndfree[j, 1]],
            c="gray",
            linestyle="dashed",
            alpha=0.2,
        )
for i, g in gQfree_dist.items():
    for j, _ in g:
        if j == len(QfulRndfree):
            # print("just expression of no edge, skipping")
            continue
        ax.plot(
            [QfulRndfree[i, 0], QfulRndfree[j, 0]],
            [QfulRndfree[i, 1], QfulRndfree[j, 1]],
            c="blue",
            linestyle="dashed",
            alpha=0.5,
        )
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("C-space dataset")
ax.set_aspect("equal")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
# ax.legend()


# local space
qs = np.array([0.15, 0.60]).reshape(-1, 1)
qg = np.array([2.5, 1.5]).reshape(-1, 1)
cmin = np.linalg.norm(qg - qs)
cMaxguess = 1.5 * cmin
Xinf_inside = informed_sampling_bulk(qs, qg, cMaxguess, 100)
Xinf_surf = informed_surface_sampling_bulk(qs, qg, cMaxguess, 100)
Xinf = np.vstack((Xinf_inside, Xinf_surf))
print("Random informed:", Xinf.shape)

# separate free and coll samples
Xinfcheck = np.zeros((Xinf.shape[0], 1))
for i in range(Xinf.shape[0]):
    q = Xinf[i, :].reshape(-1, 1)
    in_collision = collision_check(q)
    if in_collision:
        Xinfcheck[i, 0] = 1
    else:
        Xinfcheck[i, 0] = 0
Xinffree = Xinf[Xinfcheck.flatten() == 0]
Xinfcoll = Xinf[Xinfcheck.flatten() == 1]
print(f"Xinffree: {Xinffree.shape}, Xinfcoll: {Xinfcoll.shape}")

# build graphs
gXinf_no_dist, tXinf_no_dist = build_graph(Xinffree, k=10)
knnradius_og_inf = 0.5
gXinf_dist, tXinf_dist = build_graph(Xinffree, k=10, dist_thres=knnradius_og_inf)

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(datasetcoll[:, 0], datasetcoll[:, 1], c="brown", s=5)
ax2.scatter(Xinffree[:, 0], Xinffree[:, 1], s=5, c="y", label="informed free")
ax2.scatter(Xinfcoll[:, 0], Xinfcoll[:, 1], s=5, c="k", label="informed coll")
ax2.scatter(qs[0], qs[1], c="green", marker="*", s=200, label="start")
ax2.scatter(qg[0], qg[1], c="red", marker="*", s=200, label="goal")
for i, g in gXinf_no_dist.items():
    for j, _ in g:
        if j == len(Xinffree):
            # print("just expression of no edge, skipping")
            continue
        ax2.plot(
            [Xinffree[i, 0], Xinffree[j, 0]],
            [Xinffree[i, 1], Xinffree[j, 1]],
            c="gray",
            linestyle="dashed",
            alpha=0.2,
        )
for i, g in gXinf_dist.items():
    for j, _ in g:
        if j == len(Xinffree):
            # print("just expression of no edge, skipping")
            continue
        ax2.plot(
            [Xinffree[i, 0], Xinffree[j, 0]],
            [Xinffree[i, 1], Xinffree[j, 1]],
            c="blue",
            linestyle="dashed",
            alpha=0.5,
        )
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_title("C-space dataset")
ax2.set_aspect("equal")
ax2.set_xlim([-np.pi, np.pi])
ax2.set_ylim([-np.pi, np.pi])
# ax.legend()
plt.show()


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


def dijkstra_all_paths_online_knn(nodes, root, kdtree, dist_thres):
    dist = {i: float("inf") for i in range(len(nodes))}
    parent = {i: None for i in range(len(nodes))}

    dist[root] = 0
    pq = [(0, root)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        # Query k-nearest neighbors of u
        dists, idxs = kdtree.query(
            nodes[u], k=kdtree.n, distance_upper_bound=dist_thres
        )
        for v_idx, weight in zip(idxs, dists):
            if v_idx == len(nodes) or np.isinf(weight):
                continue
            new_dist = current_dist + weight
            if new_dist < dist[v_idx]:
                dist[v_idx] = new_dist
                parent[v_idx] = u
                heapq.heappush(pq, (new_dist, v_idx))

    return dist, parent


def dijkstra_single_path_online_knn(nodes, root, target, kdtree, dist_thres):
    dist = {i: float("inf") for i in range(len(nodes))}
    parent = {i: None for i in range(len(nodes))}

    dist[root] = 0
    pq = [(0, root)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        if u == target:
            break

        # Query k-nearest neighbors of u
        dists, idxs = kdtree.query(
            nodes[u], k=kdtree.n, distance_upper_bound=dist_thres
        )
        for v_idx, weight in zip(idxs, dists):
            if v_idx == len(nodes) or np.isinf(weight):
                continue
            new_dist = current_dist + weight
            if new_dist < dist[v_idx]:
                dist[v_idx] = new_dist
                parent[v_idx] = u
                heapq.heappush(pq, (new_dist, v_idx))

    return dist, parent


def reconstruct_path(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# approximate shortest path in global space
graph, tree = build_graph(QfulRndfree, k=10, dist_thres=0.5)
_, near_qsi = tree.query(qs.flatten(), k=1)
_, near_to_goal_id = tree.query(qg.flatten(), k=1)
print(f"Nearest to start: {near_qsi}")
print(f"Nearest to goal: {near_to_goal_id}")
_, parent = dijkstra_all_paths(graph, root=near_qsi)
path = reconstruct_path(parent, near_to_goal_id)
_, parent_online = dijkstra_all_paths_online_knn(
    QfulRndfree, root=near_qsi, kdtree=tree, dist_thres=0.5
)
path_online = reconstruct_path(parent_online, near_to_goal_id)
_, parent_online_single = dijkstra_single_path_online_knn(
    QfulRndfree, root=near_qsi, target=near_to_goal_id, kdtree=tree, dist_thres=0.5
)
path_online_single = reconstruct_path(parent_online_single, near_to_goal_id)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(datasetcoll[:, 0], datasetcoll[:, 1], c="brown", s=5)
ax.scatter(QfulRndfree[:, 0], QfulRndfree[:, 1], c="g", s=25, label="Qf_free")
ax.scatter(QfulRndcoll[:, 0], QfulRndcoll[:, 1], c="k", s=25, label="Qf_coll")
ax.plot(QfulRndfree[path, 0], QfulRndfree[path, 1], "r-", label="shrtst path")
ax.plot(
    QfulRndfree[path_online, 0],
    QfulRndfree[path_online, 1],
    "b-",
    label="online path",
)
ax.plot(
    QfulRndfree[path_online_single, 0],
    QfulRndfree[path_online_single, 1],
    "g-",
    label="online single path",
)
ax.scatter(qs[0], qs[1], c="green", marker="*", s=100, label="start")
ax.scatter(qg[0], qg[1], c="red", marker="*", s=100, label="goal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_aspect("equal")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.legend()
plt.show()
