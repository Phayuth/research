import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import heapq

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def build_graph(points, k, dist_threshold=np.inf):
    tree = KDTree(points)
    graph = {i: [] for i in range(len(points))}
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k + 1, distance_upper_bound=dist_threshold)
        for j, d in zip(idx[1:], dists[1:]):
            graph[i].append((j, float(d)))
    return graph, tree


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


class RTSPLazyEllipsoid:

    def __init__(self, collsion_checker, lmts=None):
        self.collision_checker = collsion_checker
        self.lmts = lmts

    def print_info(self):
        pass

    def estimate_shortest_path(self, qs, qg):
        cmin = np.linalg.norm(qg - qs)
        num_step = 5
        cMaxguesses = np.linspace(cmin, 1.5 * cmin, num=num_step)
        num = 100

        Xin = np.empty((num * num_step, 2))
        Xsf = np.empty((num * num_step, 2))
        for i, cMaxguess in enumerate(cMaxguesses):
            Xinside = informed_sampling_bulk(qs, qg, cMaxguess, num)
            Xsurf = informed_surface_sampling_bulk(qs, qg, cMaxguess, num)
            Xin[i * num : (i + 1) * num] = Xinside
            Xsf[i * num : (i + 1) * num] = Xsurf
        Qrand = np.vstack((Xin, Xsf))
        Qrandstat = np.zeros((Qrand.shape[0], 1))
        for i in range(Qrand.shape[0]):
            q = Qrand[i, :]
            in_collision = self.collision_checker(q)
            if in_collision:
                Qrandstat[i, 0] = 1
            else:
                Qrandstat[i, 0] = 0
        Qrandfree = Qrand[Qrandstat.flatten() == 0]

        # build some KD-trees for nearest neighbor queries
        Qrandfree = np.vstack((qs.flatten(), qg.flatten(), Qrandfree))
        graph, kdt = build_graph(Qrandfree, k=10)

        rootid = 0
        goalid = 1
        # solve shortest paths from root to all nodes
        _, parents = dijkstra_all_paths(graph, root=rootid)
        qpath_ids = reconstruct_path(parents, goalid)
        qpath = Qrandfree[qpath_ids]
        cost = get_path_cost(qpath)
        return qpath, cost, Qrandfree


if __name__ == "__main__":
    from paper_sequential_planner.scripts.geometric_ellipse import *
    from paper_sequential_planner.experiments.env_planarrr import *

    robot = PlanarRR()
    scene = RobotScene(robot, None)
    cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
    estor = RTSPLazyEllipsoid(scene.collision_checker)
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

    # qs = Q[0].reshape(-1, 1)
    # qg = Q[6].reshape(-1, 1)
    # cmin = np.linalg.norm(qg - qs)
    # cMaxguess = 1.5 * cmin
    # la0 = cmin / 2
    # sa0 = 0
    # la1 = cMaxguess / 2
    # sa1 = cMaxguess / 2
    # la2 = cmin / 2
    # sa2 = cmin / 2
    # la3 = np.sqrt(cMaxguess**2 - cmin**2) / 2
    # sa3 = cMaxguess / 2
    # la4 = cmin / 2
    # sa4 = np.sqrt(cMaxguess**2 - cmin**2) / 2
    # la5 = np.sqrt(cMaxguess**2 - cmin**2) / 2
    # sa5 = cmin / 2

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    i = 0
    j = 1
    imax = Q.shape[0]
    (path,) = ax.plot(
        [],
        [],
        "g-",
        linewidth=2,
        label="Estimated Shortest Path",
    )
    (qs_pt,) = ax.plot(
        [],
        [],
        marker="o",
        color="blue",
        label="Start",
    )
    (qg_pt,) = ax.plot(
        [],
        [],
        marker="o",
        color="green",
        label="Goal",
    )
    (xfree_pts,) = ax.plot(
        [],
        [],
        "kx",
        alpha=0.5,
        markersize=1,
        label="Free Samples",
    )

    def viewer(event):
        global i, imax
        if event.key == "up":
            i = (i + 1) % imax
        elif event.key == "down":
            i = (i - 1) % imax
        qs = Q[i % imax].reshape(-1, 1)
        qg = Q[(i + 6) % imax].reshape(-1, 1)
        qpath, cost, xfree = estor.estimate_shortest_path(qs, qg)
        print(f"==>> qpath: \n{qpath}")
        print(f"==>> cost: \n{cost}")
        path.set_data(qpath[:, 0], qpath[:, 1])
        qs_pt.set_data(qs[0], qs[1])
        qg_pt.set_data(qg[0], qg[1])
        xfree_pts.set_data(xfree[:, 0], xfree[:, 1])
        plt.draw()

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.connect("key_press_event", viewer)
    plt.show()
