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


def method_0000():
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
                ax.plot([Q[i, 0], Q[j, 0]], [Q[i, 1], Q[j, 1]], "g-", alpha=0.5)
            else:
                ax.plot([Q[i, 0], Q[j, 0]], [Q[i, 1], Q[j, 1]], "r-", alpha=0.5)
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


def method1():
    cmin = np.linalg.norm(qg - qs)
    cMaxguess = 1.5 * cmin

    Xinf = informed_sampling_bulk(qs, qg, cMaxguess, 1000)
    Xinf_surf = informed_surface_sampling_bulk(qs, qg, cMaxguess, 1000)

    la0 = cmin / 2
    sa0 = 0
    Xb0 = custom_surface_sampling(qs, qg, la0, sa0, 1000)
    Xi0 = custom_inside_sampling(qs, qg, la0, sa0, 1000)

    la1 = cMaxguess / 2
    sa1 = cMaxguess / 2
    Xb1 = custom_surface_sampling(qs, qg, la1, sa1, 1000)
    Xi1 = custom_inside_sampling(qs, qg, la1, sa1, 1000)

    la2 = cmin / 2
    sa2 = cmin / 2
    Xb2 = custom_surface_sampling(qs, qg, la2, sa2, 1000)
    Xi2 = custom_inside_sampling(qs, qg, la2, sa2, 1000)

    la3 = np.sqrt(cMaxguess**2 - cmin**2) / 2
    sa3 = cMaxguess / 2
    Xb3 = custom_surface_sampling(qs, qg, la3, sa3, 1000)
    Xi3 = custom_inside_sampling(qs, qg, la3, sa3, 1000)

    la4 = cmin / 2
    sa4 = np.sqrt(cMaxguess**2 - cmin**2) / 2
    Xb4 = custom_surface_sampling(qs, qg, la4, sa4, 1000)
    Xi4 = custom_inside_sampling(qs, qg, la4, sa4, 1000)

    la5 = np.sqrt(cMaxguess**2 - cmin**2) / 2
    sa5 = cmin / 2
    Xb5 = custom_surface_sampling(qs, qg, la5, sa5, 1000)
    Xi5 = custom_inside_sampling(qs, qg, la5, sa5, 1000)

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    ax.scatter(Xinf_surf[:, 0], Xinf_surf[:, 1], s=5, c="b", label="informed")
    ax.scatter(Xi0[:, 0], Xi0[:, 1], s=5, c="c", label="inside0")
    ax.scatter(Xb1[:, 0], Xb1[:, 1], s=5, c="g", label="equals")
    ax.scatter(Xb2[:, 0], Xb2[:, 1], s=5, c="r", label="case1")
    ax.scatter(Xb3[:, 0], Xb3[:, 1], s=5, c="m", label="case2")
    ax.scatter(Xb4[:, 0], Xb4[:, 1], s=5, c="c", label="case3")
    ax.scatter(Xb5[:, 0], Xb5[:, 1], s=5, c="y", label="case4")
    ax.scatter(qs[0], qs[1], s=50, c="k", marker="x")
    ax.scatter(qg[0], qg[1], s=50, c="k", marker="x")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
    plt.show()


def method3(qs=qs, qg=qg):
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
    print(f"Shortest path from root to goal (id {goalid}): {pathgoal}")
    print(f"Path length: {distances[goalid]:.3f}")
    print("Path coordinates:\n", qpath)

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3, alpha=0.1)
    for i in range(num_guess):
        cMaxguess = cMaxguesses[i]
        idx_start = i * num_per_guess
        idx_end = (i + 1) * num_per_guess
        ax.scatter(
            Xinf_surfaces[idx_start:idx_end, 0],
            Xinf_surfaces[idx_start:idx_end, 1],
            s=5,
            color=plt.cm.viridis(i / num_guess),
            label=f"surface {i+1}, c_max = {cMaxguess:.2f}",
        )
        ax.scatter(
            Xinf_insides[idx_start:idx_end, 0],
            Xinf_insides[idx_start:idx_end, 1],
            s=5,
            color=plt.cm.viridis(i / num_guess),
            alpha=0.5,
        )
    # # ax.plot(qpath34[:, 0], qpath34[:, 1], "r-o", linewidth=2, label="planned path")
    ax.scatter(qs[0], qs[1], s=50, c="g", marker="s", label="Start")
    ax.scatter(qg[0], qg[1], s=50, c="r", marker="s", label="Goal")
    ax.plot(qpath[:, 0], qpath[:, 1], "r-o", linewidth=2, label="Dijkstra path")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
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


class RTSPLazyEllipsoid:

    def __init__(self):
        pass


if __name__ == "__main__":
    # method_0000()
    # method1()
    # method3()
    # allnode_RGG_sparse()
    allnode_RGG_sparse_robot()
    # allnode_RGG_sparse_3d()
