import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import qmc
from paper_sequential_planner.scripts.geometric_torus import find_alt_config2
import tqdm

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


def graph_mid_point_collision_prune(graph, points, collision_checker):
    new_graph = {i: [] for i in graph}
    for u in graph:
        for v, d_uv in graph[u]:
            # d = np.linalg.norm(points[u] - points[v])
            mid_point = (points[u] + points[v]) / 2
            if not collision_checker(mid_point):
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
    QfulRndcoll = QfulRnd[QfulRndcheck.flatten() == 1]
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


class RTSPLazyPRMEstimator:

    def __init__(self, collision_checker, lmts=None):
        self.collision_checker = collision_checker
        self.lmts = lmts
        self.Qrandfree = None
        self.Qrandcols = None
        self.kdt = None
        self.graph = None
        # self.samples(initial_samples)

    def print_info(self):
        print(f"==>> Qrandfree.shape: {self.Qrandfree.shape}")
        print(f"==>> Qrandcols.shape: {self.Qrandcols.shape}")
        print(f"==>> kdtree.shape: {self.kdt.n}")
        print(f"==>> graph.nodes: {len(self.graph)}")

    def samples(self, num_samples):
        Qrand = np.random.uniform(
            self.lmts[:, 0],
            self.lmts[:, 1],
            size=(num_samples, self.lmts.shape[0]),
        )
        Qrandstat = np.zeros((num_samples, 1))
        for i in range(num_samples):
            q = Qrand[i, :]
            in_collision = self.collision_checker(q)
            if in_collision:
                Qrandstat[i, 0] = 1
            else:
                Qrandstat[i, 0] = 0
        if self.Qrandfree is None:
            self.Qrandfree = Qrand[Qrandstat.flatten() == 0]
            self.Qrandcols = Qrand[Qrandstat.flatten() == 1]
        else:
            self.Qrandfree = np.vstack(
                [self.Qrandfree, Qrand[Qrandstat.flatten() == 0]]
            )
            self.Qrandcols = np.vstack(
                [self.Qrandcols, Qrand[Qrandstat.flatten() == 1]]
            )
        # rebuild kdtree, graph with new samples
        self.kdt = KDTree(self.Qrandfree)
        self.graph = self.build_graph(k=10, dist_thres=0.5)

    def samples_sparse(self, num_samples, eps):
        Qrand = np.random.uniform(
            self.lmts[:, 0],
            self.lmts[:, 1],
            size=(num_samples, self.lmts.shape[0]),
        )
        Qrandstat = np.zeros((num_samples, 1))
        for i in range(num_samples):
            q = Qrand[i, :]
            in_collision = self.collision_checker(q)
            if in_collision:
                Qrandstat[i, 0] = 1
            else:
                Qrandstat[i, 0] = 0
        self.Qrandfree = Qrand[Qrandstat.flatten() == 0]
        self.Qrandcols = Qrand[Qrandstat.flatten() == 1]
        self.Qrandfree = sparsify_nodes(self.Qrandfree, eps)

        self.kdt = KDTree(self.Qrandfree)
        self.graph = self.build_graph(k=5)
        self.graph = prune_edges_triangle(self.graph, self.Qrandfree, delta=0.1)
        self.graph = graph_mid_point_collision_prune(
            self.graph, self.Qrandfree, self.collision_checker
        )

    def build_graph(self, k, dist_thres=np.inf):
        graph = {i: [] for i in range(len(self.Qrandfree))}
        for i, p in tqdm.tqdm(
            enumerate(self.Qrandfree),
            total=len(self.Qrandfree),
            desc="Building graph",
        ):
            # for i, p in enumerate(self.Qrandfree):
            dists, idx = self.kdt.query(p, k + 1, distance_upper_bound=dist_thres)
            for j, d in zip(idx[1:], dists[1:]):
                if j == len(self.Qrandfree) or np.isinf(d):
                    continue
                graph[i].append((int(j), float(d)))
        return graph

    def estimate_shortest_path(self, qs, qg):
        _, near_qs_id = self.kdt.query(qs, k=1)
        _, near_qg_id = self.kdt.query(qg, k=1)

        # dijkstra mode : single path from root to target, online knn query
        _, parent = dijkstra_single_path_online_knn(
            nodes=self.Qrandfree,
            root=near_qs_id,
            target=near_qg_id,
            kdtree=self.kdt,
            dist_thres=0.5,
        )
        pathid = reconstruct_path(parent, near_qg_id)
        pathq_mid = self.Qrandfree[pathid]
        pathq_full = np.vstack([qs, pathq_mid, qg])
        cost = get_path_cost(pathq_full)
        return pathq_full, cost

    def estimate_shortest_path_bulk(self, qs, qgs):
        _, near_qs_id = self.kdt.query(qs, k=1)

        near_qgs_ids = []
        for qg in qgs:
            _, near_qg_id = self.kdt.query(qg, k=1)
            near_qgs_ids.append(near_qg_id)

        targets = set(near_qgs_ids)
        _, parent = dijkstra_bulk_paths(
            self.graph, root=near_qs_id, targets=targets
        )

        paths = []
        costs = []
        for i, near_qg_id in enumerate(near_qgs_ids):
            pathid = reconstruct_path(parent, near_qg_id)
            pathq_mid = self.Qrandfree[pathid]
            pathq_full = np.vstack([qs, pathq_mid, qgs[i]])
            cost = get_path_cost(pathq_full)
            paths.append(pathq_full)
            costs.append(cost)

        return paths, costs


class RTSPLazyPRMEstimatorExtended(RTSPLazyPRMEstimator):

    def __init__(self, collision_checker, lmts=None):
        super().__init__(collision_checker, lmts)

    def samples(self, num_samples):
        Qrand = np.random.uniform(
            self.lmts[:, 0],
            self.lmts[:, 1],
            size=(num_samples, self.lmts.shape[0]),
        )
        Qrandstat = np.zeros((num_samples, 1))
        for i in range(num_samples):
            q = Qrand[i, :]
            in_collision = self.collision_checker(q)
            if in_collision:
                Qrandstat[i, 0] = 1
            else:
                Qrandstat[i, 0] = 0

        Qrandfree = Qrand[Qrandstat.flatten() == 0]
        Qrandcols = Qrand[Qrandstat.flatten() == 1]

        numext_per_q = 4
        Qrandfree_ext = np.zeros(
            (Qrandfree.shape[0] * numext_per_q, Qrandfree.shape[1])
        )
        Qrandcols_ext = np.zeros(
            (Qrandcols.shape[0] * numext_per_q, Qrandcols.shape[1])
        )
        for i in range(Qrandfree.shape[0]):
            q = Qrandfree[i, :]
            q_ext = find_alt_config2(q, self.lmts, filterOriginalq=False)
            Qrandfree_ext[i * numext_per_q : (i + 1) * numext_per_q, :] = q_ext
        for i in range(Qrandcols.shape[0]):
            q = Qrandcols[i, :]
            q_ext = find_alt_config2(q, self.lmts, filterOriginalq=False)
            Qrandcols_ext[i * numext_per_q : (i + 1) * numext_per_q, :] = q_ext

        if self.Qrandfree is None:
            self.Qrandfree = Qrandfree_ext
            self.Qrandcols = Qrandcols_ext
        else:
            self.Qrandfree = np.vstack([self.Qrandfree, Qrandfree_ext])
            self.Qrandcols = np.vstack([self.Qrandcols, Qrandcols_ext])

        # rebuild kdtree, graph with new samples
        self.kdt = KDTree(self.Qrandfree)
        self.graph = self.build_graph(k=10, dist_thres=0.5)

    def samples_sparse(self, num_samples, eps):
        Qrand = np.random.uniform(
            self.lmts[:, 0],
            self.lmts[:, 1],
            size=(num_samples, self.lmts.shape[0]),
        )
        Qrandstat = np.zeros((num_samples, 1))
        for i in range(num_samples):
            q = Qrand[i, :]
            in_collision = self.collision_checker(q)
            if in_collision:
                Qrandstat[i, 0] = 1
            else:
                Qrandstat[i, 0] = 0

        Qrandfree = Qrand[Qrandstat.flatten() == 0]
        Qrandcols = Qrand[Qrandstat.flatten() == 1]
        Qrandfree = sparsify_nodes(Qrandfree, eps)

        numext_per_q = 4  # to change to different number of space
        Qrandfree_ext = np.zeros(
            (Qrandfree.shape[0] * numext_per_q, Qrandfree.shape[1])
        )
        Qrandcols_ext = np.zeros(
            (Qrandcols.shape[0] * numext_per_q, Qrandcols.shape[1])
        )
        for i in range(Qrandfree.shape[0]):
            q = Qrandfree[i, :]
            q_ext = find_alt_config2(q, self.lmts, filterOriginalq=False)
            Qrandfree_ext[i * numext_per_q : (i + 1) * numext_per_q, :] = q_ext
        for i in range(Qrandcols.shape[0]):
            q = Qrandcols[i, :]
            q_ext = find_alt_config2(q, self.lmts, filterOriginalq=False)
            Qrandcols_ext[i * numext_per_q : (i + 1) * numext_per_q, :] = q_ext

        self.Qrandfree = Qrandfree_ext
        self.Qrandcols = Qrandcols_ext

        self.kdt = KDTree(self.Qrandfree)
        self.graph = self.build_graph(k=5)
        self.graph = prune_edges_triangle(self.graph, self.Qrandfree, delta=0.1)
        self.graph = graph_mid_point_collision_prune(
            self.graph, self.Qrandfree, self.collision_checker
        )


class RTSPLazyPRMPoissonDisk(RTSPLazyPRMEstimator):

    def __init__(self, collision_checker, lmts=None):
        super().__init__(collision_checker, lmts)

    def samples_sparse(self, num_samples=500):
        dof = self.lmts.shape[0]
        radius = 0.05
        engine = qmc.PoissonDisk(d=dof, radius=radius)
        Qrand = engine.random(num_samples)
        Qrand = Qrand * 2 * np.pi - np.pi  # scale to [-pi, pi]
        numfilled = Qrand.shape[0]
        Qrandstat = np.zeros((numfilled, 1))
        for i in tqdm.tqdm(range(numfilled), desc="Collision checking samples"):
            q = Qrand[i, :]
            in_collision = self.collision_checker(q)
            if in_collision:
                Qrandstat[i, 0] = 1
            else:
                Qrandstat[i, 0] = 0
        self.Qrandfree = Qrand[Qrandstat.flatten() == 0]
        self.Qrandcols = Qrand[Qrandstat.flatten() == 1]

        self.kdt = KDTree(self.Qrandfree)
        self.graph = self.build_graph(k=5)
        self.graph = prune_edges_triangle(self.graph, self.Qrandfree, delta=0.1)
        self.graph = graph_mid_point_collision_prune(
            self.graph, self.Qrandfree, self.collision_checker
        )


class RTSPLazyPRMPoissonDiskExtended(RTSPLazyPRMEstimator):

    def __init__(self, collision_checker, lmts=None):
        super().__init__(collision_checker, lmts)

    def find_numext_per_q(self, q):
        q_ext = find_alt_config2(q, self.lmts)
        return q_ext.shape[0]

    def samples_sparse(self, num_samples=500):
        dof = self.lmts.shape[0]
        radius = 0.05
        engine = qmc.PoissonDisk(d=dof, radius=radius)
        Qrand = engine.random(num_samples)
        Qrand = Qrand * 2 * np.pi - np.pi  # scale to [-pi, pi]
        numfilled = Qrand.shape[0]
        Qrandstat = np.zeros((numfilled, 1))
        for i in tqdm.tqdm(range(numfilled), desc="Collision checking samples"):
            q = Qrand[i, :]
            in_collision = self.collision_checker(q)
            if in_collision:
                Qrandstat[i, 0] = 1
            else:
                Qrandstat[i, 0] = 0
        Qrandfree = Qrand[Qrandstat.flatten() == 0]
        Qrandcols = Qrand[Qrandstat.flatten() == 1]

        numext_per_q = self.find_numext_per_q(Qrandfree[0])
        Qrandfree_ext = np.zeros(
            (Qrandfree.shape[0] * numext_per_q, Qrandfree.shape[1])
        )
        Qrandcols_ext = np.zeros(
            (Qrandcols.shape[0] * numext_per_q, Qrandcols.shape[1])
        )
        for i in range(Qrandfree.shape[0]):
            q = Qrandfree[i, :]
            q_ext = find_alt_config2(q, self.lmts, filterOriginalq=False)
            Qrandfree_ext[i * numext_per_q : (i + 1) * numext_per_q, :] = q_ext
        for i in range(Qrandcols.shape[0]):
            q = Qrandcols[i, :]
            q_ext = find_alt_config2(q, self.lmts, filterOriginalq=False)
            Qrandcols_ext[i * numext_per_q : (i + 1) * numext_per_q, :] = q_ext

        self.Qrandfree = Qrandfree_ext
        self.Qrandcols = Qrandcols_ext

        # tecnically we can just build graph on small space and duplicate to ext
        # we then repair the edges of space. but for now just build the whole thing
        self.kdt = KDTree(self.Qrandfree)
        self.graph = self.build_graph(k=5)
        self.graph = prune_edges_triangle(self.graph, self.Qrandfree, delta=0.1)
        self.graph = graph_mid_point_collision_prune(
            self.graph, self.Qrandfree, self.collision_checker
        )


def test_planarrr():
    from paper_sequential_planner.experiments.env_planarrr import (
        PlanarRR,
        RobotScene,
    )

    robot = PlanarRR()
    scene = RobotScene(robot, None)

    options = ["pi_space", "2pi_space", "pi_space_pd", "2pi_space_pd"][-1]
    if options == "pi_space":
        cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
        lmts = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
        estor = RTSPLazyPRMEstimator(
            scene.collision_checker,
            lmts=lmts,
        )
        # estor.samples(1000)
        estor.samples_sparse(1000, eps=0.05 * 2 * np.pi)
        estor.print_info()

    if options == "2pi_space":
        cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles_extended.npy"))
        lmts2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        estor = RTSPLazyPRMEstimatorExtended(
            scene.collision_checker,
            lmts=lmts2,
        )
        # estor.samples(1000)
        estor.samples_sparse(1000, eps=0.05 * 2 * np.pi)
        estor.print_info()

    if options == "pi_space_pd":
        cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
        lmts = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
        estor = RTSPLazyPRMPoissonDisk(
            scene.collision_checker,
            lmts=lmts,
        )
        estor.samples_sparse(500)
        estor.print_info()

    if options == "2pi_space_pd":
        cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles_extended.npy"))
        lmts2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        estor = RTSPLazyPRMPoissonDiskExtended(
            scene.collision_checker,
            lmts=lmts2,
        )
        estor.samples_sparse(500)
        estor.print_info()

    # search path
    qs = np.array([0.15, 0.60])
    qgs = np.array(
        [
            [2.5, 1.5],
            [1.5, 2.5],
            [-2.5, -2.5],
            [-2.5, 2.5],
            [2.5, -2.5],
            [-1.5, 0.0],
        ]
    )
    paths, costs = estor.estimate_shortest_path_bulk(qs, qgs)
    for i, (path, cost) in enumerate(zip(paths, costs)):
        print(f"Estimated path {i+1} cost: {cost}")

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    ax.scatter(
        estor.Qrandfree[:, 0],
        estor.Qrandfree[:, 1],
        color="blue",
        s=10,
        label="Free Samples",
    )
    ax.scatter(
        estor.Qrandcols[:, 0],
        estor.Qrandcols[:, 1],
        color="red",
        s=10,
        label="Collision Samples",
    )
    for i, neighbors in estor.graph.items():
        for j, _ in neighbors:
            ax.plot(
                [estor.Qrandfree[i, 0], estor.Qrandfree[j, 0]],
                [estor.Qrandfree[i, 1], estor.Qrandfree[j, 1]],
                "r--",
                alpha=0.1,
            )
    for i, path in enumerate(paths):
        ax.plot(
            path[:, 0],
            path[:, 1],
            linewidth=2,
            label=f"Estimated Shortest Path {i+1}",
        )
    ax.scatter(qs[0], qs[1], color="cyan", marker="*", s=100, label="Start")
    for i, qg in enumerate(qgs):
        ax.scatter(
            qg[0],
            qg[1],
            marker="*",
            s=100,
            label=f"Goal {i+1}",
        )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.set_title("Estimated Shortest Path in 2D Space")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def test_spatial3r():
    from paper_sequential_planner.experiments.env_spatial3r import (
        Spatial3R,
        RobotScene,
    )
    import trimesh

    robot = Spatial3R()
    scene = RobotScene(robot, None)

    lmts2 = np.array([[-np.pi, np.pi]] * 3)
    estor = RTSPLazyPRMPoissonDisk(
        scene.collision_checker,
        lmts=lmts2,
    )
    estor.samples_sparse(5000)
    # estor.print_info()

    # scene setup
    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=np.pi)
    box = trimesh.creation.box(extents=(2 * np.pi, 2 * np.pi, 2 * np.pi))
    box.visual.face_colors = [100, 150, 255, 40]
    scene.add_geometry(box)
    scene.add_geometry(axis)

    Qfree = estor.Qrandfree
    Qcoll = estor.Qrandcols
    qsid = np.random.choice(Qfree.shape[0], size=1, replace=False)
    qgsid = np.random.choice(Qfree.shape[0], size=5, replace=False)
    qs = Qfree[qsid].flatten()
    print(f"==>> qs: \n{qs}")
    qgs = Qfree[qgsid]
    print(f"==>> qgs: \n{qgs}")

    paths, costs = estor.estimate_shortest_path_bulk(qs, qgs)
    for i, (path, cost) in enumerate(zip(paths, costs)):
        print(f"Estimated path {i+1} cost: {cost}")

    Qff = trimesh.points.PointCloud(Qfree)
    Qff.visual.vertex_colors = np.tile(
        np.array([0, 255, 0, 180], dtype=np.uint8), (len(Qfree), 1)
    )
    scene.add_geometry(Qff)
    Qcc = trimesh.points.PointCloud(Qcoll)
    Qcc.visual.vertex_colors = np.tile(
        np.array([255, 0, 0, 180], dtype=np.uint8), (len(Qcoll), 1)
    )
    scene.add_geometry(Qcc)
    # for i, p in enumerate(store_path.values()):
    #     # p = np.linspace(s, g, 10)  # n, 3
    #     edges = np.column_stack((np.arange(len(p) - 1), np.arange(1, len(p))))
    #     path = trimesh.load_path(p[edges])
    #     scene.add_geometry(path)
    scene.show(point_size=6)


if __name__ == "__main__":
    # test_planarrr()
    test_spatial3r()
