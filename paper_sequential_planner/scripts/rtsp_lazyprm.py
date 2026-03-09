import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def build_graph(points, k, dist_thres=np.inf):
    kdtree = KDTree(points)
    graph = {i: [] for i in range(len(points))}
    for i, p in enumerate(points):
        dists, idx = kdtree.query(p, k + 1, distance_upper_bound=dist_thres)
        for j, d in zip(idx[1:], dists[1:]):
            if j == len(points) or np.isinf(d):
                continue
            graph[i].append((int(j), float(d)))
    return graph, kdtree


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
    """This is too much redundant computation with KNN"""
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


def get_path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        cost += np.linalg.norm(path[i + 1] - path[i])
    return cost


def separate_sample(collision_checker, Qful_snum=1000, lmts=None):
    if lmts is None:
        lmts = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
    QfulRnd = np.random.uniform(
        lmts[:, 0], lmts[:, 1], size=(Qful_snum, lmts.shape[0])
    )
    QfulRndcheck = np.zeros((Qful_snum, 1))
    for i in range(Qful_snum):
        q = QfulRnd[i, :]
        in_collision = collision_checker(q)
        if in_collision:
            QfulRndcheck[i, 0] = 1
        else:
            QfulRndcheck[i, 0] = 0
    QfulRndfree = QfulRnd[QfulRndcheck.flatten() == 0]
    print(f"==>> QfulRndfree.shape: {QfulRndfree.shape}")
    QfulRndcoll = QfulRnd[QfulRndcheck.flatten() == 1]
    print(f"==>> QfulRndcoll.shape: {QfulRndcoll.shape}")
    return QfulRndfree, QfulRndcoll


def estimate_shortest_path(qs, qg, Qfree, graph, kdtree):
    _, near_qs_id = kdtree.query(qs, k=1)
    _, near_qg_id = kdtree.query(qg, k=1)

    # dijkstra mode : all node back to root
    # _, parent = dijkstra_all_paths(graph, root=near_qs_id)

    # dijkstra mode : all node back to root, but online knn query
    # _, parent = dijkstra_all_paths_online_knn(
    #     QfulRndfree, root=near_qs_id, kdtree=kdtree, dist_thres=0.5
    # )

    # dijkstra mode : single path from root to target, online knn query
    _, parent = dijkstra_single_path_online_knn(
        nodes=Qfree,
        root=near_qs_id,
        target=near_qg_id,
        kdtree=kdtree,
        dist_thres=0.5,
    )
    pathid = reconstruct_path(parent, near_qg_id)
    pathq_mid = Qfree[pathid]
    pathq_full = np.vstack([qs, pathq_mid, qg])
    cost = get_path_cost(pathq_full)
    return pathq_full, cost


def estimate_shortest_path_bulk(qs, qgs, Qfree, graph, kdtree):
    _, near_qs_id = kdtree.query(qs, k=1)

    near_qgs_ids = []
    for qg in qgs:
        _, near_qg_id = kdtree.query(qg, k=1)
        near_qgs_ids.append(near_qg_id)

    targets = set(near_qgs_ids)
    _, parent = dijkstra_bulk_paths(graph, root=near_qs_id, targets=targets)

    paths = []
    costs = []
    for i, near_qg_id in enumerate(near_qgs_ids):
        pathid = reconstruct_path(parent, near_qg_id)
        pathq_mid = Qfree[pathid]
        pathq_full = np.vstack([qs, pathq_mid, qgs[i]])
        cost = get_path_cost(pathq_full)
        paths.append(pathq_full)
        costs.append(cost)

    return paths, costs


if __name__ == "__main__":
    from paper_sequential_planner.experiments.env_planarrr import (
        PlanarRR,
        RobotScene,
    )

    robot = PlanarRR()
    scene = RobotScene(robot, None)

    QfulRndfree, QfulRndcoll = separate_sample(scene.collision_checker)
    graph, kdtree = build_graph(QfulRndfree, k=10, dist_thres=0.5)

    qs = np.array([0.15, 0.60])
    qg = np.array([2.5, 1.5])
    pathq, cost = estimate_shortest_path(qs, qg, QfulRndfree, graph, kdtree)
    print(f"Estimated path cost: {cost}")

    fig, ax = plt.subplots()
    ax.scatter(
        QfulRndfree[:, 0],
        QfulRndfree[:, 1],
        color="blue",
        s=10,
        label="Free Samples",
    )
    ax.scatter(
        QfulRndcoll[:, 0],
        QfulRndcoll[:, 1],
        color="red",
        s=10,
        label="Collision Samples",
    )
    ax.plot(
        pathq[:, 0],
        pathq[:, 1],
        color="green",
        linewidth=2,
        label="Estimated Shortest Path",
    )
    ax.scatter(qs[0], qs[1], color="cyan", marker="*", s=100, label="Start")
    ax.scatter(qg[0], qg[1], color="magenta", marker="*", s=100, label="Goal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.set_title("Estimated Shortest Path in 2D Space")
    ax.legend()
    plt.show()

    qs = np.array([0.15, 0.60])
    qg1 = np.array([2.5, 1.5])
    qg2 = np.array([1.5, 2.5])
    qg3 = np.array([-2.5, -2.5])
    qg4 = np.array([-2.5, 2.5])
    qg5 = np.array([2.5, -2.5])
    qg6 = np.array([-1.5, 0.0])
    qgs = np.vstack([qg1, qg2, qg3, qg4, qg5, qg6])
    paths, costs = estimate_shortest_path_bulk(qs, qgs, QfulRndfree, graph, kdtree)
    for i, (path, cost) in enumerate(zip(paths, costs)):
        print(f"Estimated path {i+1} cost: {cost}")

    fig, ax = plt.subplots()
    ax.scatter(
        QfulRndfree[:, 0],
        QfulRndfree[:, 1],
        color="blue",
        s=10,
        label="Free Samples",
    )
    ax.scatter(
        QfulRndcoll[:, 0],
        QfulRndcoll[:, 1],
        color="red",
        s=10,
        label="Collision Samples",
    )
    for i, path in enumerate(paths):
        ax.plot(
            path[:, 0],
            path[:, 1],
            linewidth=2,
            label=f"Estimated Shortest Path {i+1}",
        )
    ax.scatter(qs[0], qs[1], color="cyan", marker="*", s=100, label="Start")
    ax.scatter(qg1[0], qg1[1], color="magenta", marker="*", s=100, label="Goal 1")
    ax.scatter(qg2[0], qg2[1], color="orange", marker="*", s=100, label="Goal 2")
    ax.scatter(qg3[0], qg3[1], color="purple", marker="*", s=100, label="Goal 3")
    ax.scatter(qg4[0], qg4[1], color="brown", marker="*", s=100, label="Goal 4")
    ax.scatter(qg5[0], qg5[1], color="pink", marker="*", s=100, label="Goal 5")
    ax.scatter(qg6[0], qg6[1], color="gray", marker="*", s=100, label="Goal 6")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.set_title("Estimated Shortest Path in 2D Space")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
