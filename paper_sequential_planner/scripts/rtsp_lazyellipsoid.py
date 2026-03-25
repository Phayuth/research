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


def ellipsoid_size_ratio():
    dof = 2  # degrees of freedom for the configuration space
    xStart = np.array([-0.5] * dof).reshape(-1, 1)
    xGoal = np.array([0.5] * dof).reshape(-1, 1)
    cminmultiplier = np.linspace(1, 5, num=10, endpoint=True)
    cMin = np.linalg.norm(xGoal - xStart)
    cMaxs = cminmultiplier * cMin
    perc = 100 * (cMaxs - cMin) / cMin

    for i in range(len(cMaxs)):
        print(
            f"cMax: {cMaxs[i]:.2f}, Max/Min: {cMaxs[i]/cMin:.2f} |{perc[i]:.2f}%"
        )

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xStart[0], xStart[1], marker="o", color="blue", label="Start")
    ax.plot(xGoal[0], xGoal[1], marker="o", color="green", label="Goal")
    for cMax in cMaxs:
        el = get_2d_ellipse_informed_mplpatch(xStart, xGoal, cMax)
        ax.add_patch(el)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)
    ax.legend()
    ax.set_title("Informed Sampling in 2D Ellipse")
    plt.show()


def sampling_xstartgoal_dataset_():
    dof = 2
    Rbest = 1.0
    cMax = 2 * Rbest
    cmin = 1.0
    ndata = 5
    shape = (ndata, dof + dof)
    Qcircle = np.empty(shape=(ndata, dof))
    Qrand = np.empty(shape=shape)
    for i in range(ndata):
        qcenter = sampling_circle_in_jointlimit(Rbest, dof)
        Qcircle[i, :] = qcenter.ravel()
        qs, qe = sampling_Xstartgoal(qcenter.reshape(-1, 1), Rbest, cmin, dof)
        Qrand[i, 0:dof] = qs.ravel()
        Qrand[i, dof : dof + dof] = qe.ravel()

    fig, ax = plt.subplots(1, 1)
    for i in range(ndata):
        qcenter = Qcircle[i, :].reshape(-1, 1)
        qs = Qrand[i, 0:dof].reshape(-1, 1)
        qe = Qrand[i, dof : dof + dof].reshape(-1, 1)

        ax.plot(qs[0], qs[1], marker="o", color="blue")
        ax.plot(qe[0], qe[1], marker="o", color="green")
        e = get_2d_ellipse_informed_mplpatch(qs, qe, cMax)
        ax.add_patch(e)
        c = get_2d_circle_mplpatch(qcenter, Rbest)
        ax.add_patch(c)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)
    ax.set_title("Dataset of Start and Goal Sampling in Informed Ellipses")
    plt.show()


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


class RTSPLazyEllipsoid:

    def __init__(self, collsion_checker, lmts=None):
        self.collision_checker = collsion_checker
        self.lmts = lmts

    def print_info(self):
        pass

    def init_estimate(self):
        print(f"==>> Q.shape: \n{Q.shape}")
        euc_dist = euclidean_distances(Q)
        print(f"==>> euc_dist: \n{euc_dist}")
        print(f"==>> euc_dist.shape: \n{euc_dist.shape}")

        straight_valid = np.zeros((Q.shape[0], Q.shape[0]), dtype=bool)
        for i in range(Q.shape[0]):
            for j in range(i + 1, Q.shape[0]):
                if is_edge_in_collision(Q[i], Q[j]):
                    straight_valid[i, j] = False
                    print(f"Edge between q{i+1} and q{j+1} is in collision.")
                else:
                    straight_valid[i, j] = True
                    print(f"Edge between q{i+1} and q{j+1} is free.")

        fig, ax = plt.subplots()
        ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
        for i in range(Q.shape[0]):
            ax.scatter(Q[i, 0], Q[i, 1], s=50, marker="o", label=f"q{i+1}")
        for i in range(Q.shape[0]):
            for j in range(i + 1, Q.shape[0]):
                if straight_valid[i, j]:
                    ax.plot(
                        [Q[i, 0], Q[j, 0]], [Q[i, 1], Q[j, 1]], "g-", alpha=0.5
                    )
                else:
                    ax.plot(
                        [Q[i, 0], Q[j, 0]], [Q[i, 1], Q[j, 1]], "r-", alpha=0.5
                    )
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        def onclick(event):
            if event.inaxes:
                print(f"{event.xdata:.2f}, {event.ydata:.2f}")

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.grid()
        plt.show()

    def estimate_shortest_path(self, qs, qg):
        cmin = np.linalg.norm(qg - qs)
        num_guess = 5
        cMaxguesses = np.linspace(cmin, 1.5 * cmin, num=num_guess)
        num_per_guess = 100

        Xinf_insides = np.empty((num_per_guess * num_guess, 2))
        Xinf_surfaces = np.empty((num_per_guess * num_guess, 2))
        for i, cMaxguess in enumerate(cMaxguesses):
            Xinf_inside = informed_sampling_bulk(qs, qg, cMaxguess, num_per_guess)
            Xinf_surf = informed_surface_sampling_bulk(
                qs, qg, cMaxguess, num_per_guess
            )
            Xinf_insides[i * num_per_guess : (i + 1) * num_per_guess] = Xinf_inside
            Xinf_surfaces[i * num_per_guess : (i + 1) * num_per_guess] = Xinf_surf
        Xstat = bulk_collisioncheck(Xinf_insides)
        Xfree = Xinf_insides[Xstat == 0]

        # build some KD-trees for nearest neighbor queries
        Vkdt_sg = np.vstack((qs.flatten(), qg.flatten(), Xfree))
        graph, kdt = build_graph(Vkdt_sg, k=10)

        dd, ii = kdt.query(qg.flatten(), k=10)  # warm up the KD-tree
        print(f"dd: {dd}, ii: {ii}")
        rootid = 0
        goalid = 1
        rootnode = Vkdt_sg[rootid]
        goalnode = Vkdt_sg[goalid]
        # solve shortest paths from root to all nodes
        distances, paths = dijkstra_all_paths(graph, root=rootid)
        pathgoal = paths[goalid]
        qpath = Vkdt_sg[pathgoal]
        # print(f"Shortest path from root to goal (id {goalid}): {pathgoal}")
        # print(f"Path length: {distances[goalid]:.3f}")
        # print("Path coordinates:\n", qpath)
        fig, ax = plt.subplots()
        ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
        ax.plot(qs[0], qs[1], marker="o", color="blue", label="Start")
        ax.plot(qg[0], qg[1], marker="o", color="green", label="Goal")
        ax.plot(Xfree[:, 0], Xfree[:, 1], "kx", markersize=3, label="Informed Samples")
        ax.plot(qpath[:, 0], qpath[:, 1], "g-", linewidth=2, label="Estimated Shortest Path")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        plt.show()
        return qpath, distances[goalid]


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

    # reconstruct paths
    paths = {}
    for node in graph:
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        paths[node] = path[::-1]  # root → node

    return dist, paths


def build_graph(points, k, dist_threshold=np.inf):
    tree = KDTree(points)
    graph = {i: [] for i in range(len(points))}
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k + 1, distance_upper_bound=dist_threshold)
        for j, d in zip(idx[1:], dists[1:]):
            graph[i].append((j, float(d)))
    return graph, tree


if __name__ == "__main__":
    robot = PlanarRR()
    scene = RobotScene(robot, None)
    cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))

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

    estor = RTSPLazyEllipsoid(bulk_collisioncheck)
    estor.init_estimate()
    for i in range(Q.shape[0]):
        for j in range(i + 1, Q.shape[0]):
            qs = Q[i].reshape(-1, 1)
            qg = Q[j].reshape(-1, 1)
            q, cost = estor.estimate_shortest_path(qs, qg)
            print(f"==>> q: \n{q}")
            print(f"==>> cost: \n{cost}")
