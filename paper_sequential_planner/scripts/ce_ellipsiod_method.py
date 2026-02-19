import os
import numpy as np
import matplotlib.pyplot as plt
from geometric_ellipse import *
from problem_planarrr import *
from scipy.spatial import KDTree
import heapq

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
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
cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))


def bulk_collisioncheck(X):
    print(X.shape)
    Xresult = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        q = X[i, :].reshape(-1, 1)
        best, res = scene.distance_to_obstacles(q)
        if best["distance"] <= 0:
            Xresult[i, 0] = 1  # in collision
        else:
            Xresult[i, 0] = 0  # free
    return Xresult


def is_node_in_collision(q):
    best, res = scene.distance_to_obstacles(q)
    return best["distance"] <= 0


def is_edge_in_collision(q1, q2):
    qs = np.linspace(q1.flatten(), q2.flatten(), num=10).reshape(-1, 2)
    for q in qs:
        if is_node_in_collision(q.reshape(-1, 1)):
            return True
    return False


q1 = np.array([-1.0, 2.5]).reshape(-1, 1)
q2 = np.array([1.0, 2.5]).reshape(-1, 1)
q3 = np.array([0.15, 0.60]).reshape(-1, 1)
q4 = np.array([2.5, 1.5]).reshape(-1, 1)
q5 = np.array([-2.5, -1.5]).reshape(-1, 1)
q6 = np.array([2.40, -0.4]).reshape(-1, 1)
q7 = np.array([-2.0, 2.5]).reshape(-1, 1)
q8 = np.array([1.0, -2.0]).reshape(-1, 1)
q9 = np.array([-3.0, 0.0]).reshape(-1, 1)
q10 = np.array([-3.0, 2.5]).reshape(-1, 1)

qs = q3
qg = q4
qpath34 = np.array(
    [
        0.15,
        0.60,
        0.06,
        0.86,
        -0.02,
        1.16,
        -0.04,
        1.50,
        -0.04,
        1.76,
        0.04,
        1.91,
        0.24,
        2.04,
        0.72,
        2.12,
        1.12,
        2.03,
        1.31,
        1.94,
        1.61,
        1.85,
        1.90,
        1.75,
        2.22,
        1.59,
        2.5,
        1.5,
    ]
).reshape(-1, 2)


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

    # Xcol_bulk = bulk_collisioncheck(Xinf_surf)
    # print(Xcol_bulk.shape)
    # print(Xcol_bulk)

    qqq = np.array([-0.75, 3.08]).reshape(-1, 1)
    B1 = bezier_curve(qs, qqq, qg, 100)
    print(B1.shape)
    B2 = three_point_path(qs, qqq, qg, 100)
    print(B2.shape)

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    ax.scatter(Xinf_surf[:, 0], Xinf_surf[:, 1], s=5, c="b", label="informed")
    ax.scatter(Xi0[:, 0], Xi0[:, 1], s=5, c="c", label="inside0")
    ax.scatter(Xb1[:, 0], Xb1[:, 1], s=5, c="g", label="equals")
    ax.scatter(Xb2[:, 0], Xb2[:, 1], s=5, c="r", label="case1")
    ax.scatter(Xb3[:, 0], Xb3[:, 1], s=5, c="m", label="case2")
    ax.scatter(Xb4[:, 0], Xb4[:, 1], s=5, c="c", label="case3")
    ax.scatter(Xb5[:, 0], Xb5[:, 1], s=5, c="y", label="case4")
    ax.plot(B1[0, :], B1[1, :], "k-", linewidth=2, label="bezier")
    ax.plot(B2[0, :], B2[1, :], "k--", linewidth=2, label="3-point path")
    ax.plot(qqq[0], qqq[1], "kx", markersize=10, label="control point")
    ax.plot(qpath34[:, 0], qpath34[:, 1], "r-o", linewidth=2, label="planned path")
    ax.scatter(qs[0], qs[1], s=50, c="k", marker="x")
    ax.scatter(qg[0], qg[1], s=50, c="k", marker="x")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)

    def onclick(event):
        if event.inaxes:
            print(f"{event.xdata:.2f}, {event.ydata:.2f}")

    fig.canvas.mpl_connect("button_press_event", onclick)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
    plt.show()


def method2():
    cmin = np.linalg.norm(qg - qs)
    # cMaxguess = 1.5 * cmin
    numguess = 5
    cMaxguesses = np.linspace(cmin, 1.5 * cmin, num=numguess)
    sampler_per_guess = 100

    Xinf_insides = np.empty((sampler_per_guess * numguess, 2))
    Xinf_surfaces = np.empty((sampler_per_guess * numguess, 2))
    for i, cMaxguess in enumerate(cMaxguesses):
        Xinf_inside = informed_sampling_bulk(qs, qg, cMaxguess, sampler_per_guess)
        Xinf_surf = informed_surface_sampling_bulk(
            qs, qg, cMaxguess, sampler_per_guess
        )
        Xinf_insides[i * sampler_per_guess : (i + 1) * sampler_per_guess] = (
            Xinf_inside
        )
        Xinf_surfaces[i * sampler_per_guess : (i + 1) * sampler_per_guess] = (
            Xinf_surf
        )

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3, alpha=0.1)
    for i in range(numguess):
        cMaxguess = cMaxguesses[i]
        idx_start = i * sampler_per_guess
        idx_end = (i + 1) * sampler_per_guess
        ax.scatter(
            Xinf_surfaces[idx_start:idx_end, 0],
            Xinf_surfaces[idx_start:idx_end, 1],
            s=5,
            color=plt.cm.viridis(i / numguess),
            label=f"surface {i+1}, c_max = {cMaxguess:.2f}",
        )
        ax.scatter(
            Xinf_insides[idx_start:idx_end, 0],
            Xinf_insides[idx_start:idx_end, 1],
            s=5,
            color=plt.cm.viridis(i / numguess),
            alpha=0.5,
        )
    ax.plot(qpath34[:, 0], qpath34[:, 1], "r-o", linewidth=2, label="planned path")
    ax.scatter(qs[0], qs[1], s=50, c="k", marker="x")
    ax.scatter(qg[0], qg[1], s=50, c="k", marker="x")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
    plt.show()


def method3():
    cmin = np.linalg.norm(qg - qs)
    # cMaxguess = 1.5 * cmin
    numguess = 5
    cMaxguesses = np.linspace(cmin, 1.5 * cmin, num=numguess)
    sampler_per_guess = 100

    Xinf_insides = np.empty((sampler_per_guess * numguess, 2))
    Xinf_surfaces = np.empty((sampler_per_guess * numguess, 2))
    for i, cMaxguess in enumerate(cMaxguesses):
        Xinf_inside = informed_sampling_bulk(qs, qg, cMaxguess, sampler_per_guess)
        Xinf_surf = informed_surface_sampling_bulk(
            qs, qg, cMaxguess, sampler_per_guess
        )
        Xinf_insides[i * sampler_per_guess : (i + 1) * sampler_per_guess] = (
            Xinf_inside
        )
        Xinf_surfaces[i * sampler_per_guess : (i + 1) * sampler_per_guess] = (
            Xinf_surf
        )

    Xfree = np.array(
        [x for x in Xinf_insides if not is_node_in_collision(x.reshape(-1, 1))]
    )

    # build some KD-trees for nearest neighbor queries
    Vkdt_sg = np.vstack((qs.flatten(), qg.flatten(), Xfree))
    graph = build_graph(Vkdt_sg, k=10)

    rootid = 0
    goalid = 4
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
    for i in range(numguess):
        cMaxguess = cMaxguesses[i]
        idx_start = i * sampler_per_guess
        idx_end = (i + 1) * sampler_per_guess
        ax.scatter(
            Xinf_surfaces[idx_start:idx_end, 0],
            Xinf_surfaces[idx_start:idx_end, 1],
            s=5,
            color=plt.cm.viridis(i / numguess),
            label=f"surface {i+1}, c_max = {cMaxguess:.2f}",
        )
        ax.scatter(
            Xinf_insides[idx_start:idx_end, 0],
            Xinf_insides[idx_start:idx_end, 1],
            s=5,
            color=plt.cm.viridis(i / numguess),
            alpha=0.5,
        )
    ax.plot(qpath34[:, 0], qpath34[:, 1], "r-o", linewidth=2, label="planned path")
    ax.scatter(qs[0], qs[1], s=50, c="k", marker="x")
    ax.scatter(qg[0], qg[1], s=50, c="k", marker="x")
    ax.plot(qpath[:, 0], qpath[:, 1], "k-", linewidth=2, label="Dijkstra path")
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


def build_graph(points, k):
    tree = KDTree(points)
    graph = {i: [] for i in range(len(points))}
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k + 1)  # include itself
        for j, d in zip(idx[1:], dists[1:]):
            graph[i].append((j, float(d)))
    return graph


def allnode_RGG():
    points = np.random.rand(100, 2)
    k = 5

    graph = build_graph(points, k)
    rootid = 0
    goalid = 1
    rootnode = points[rootid]
    goalnode = points[goalid]
    # solve shortest paths from root to all nodes
    distances, paths = dijkstra_all_paths(graph, root=rootid)
    print(distances)
    pathgoal = paths[goalid]
    qpath = points[pathgoal]
    print(f"Shortest path from root to goal (id {goalid}): {pathgoal}")
    print(f"Path length: {distances[goalid]:.3f}")
    print("Path coordinates:\n", qpath)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=50, c="b", label="Nodes")
    ax.scatter(
        rootnode[0], rootnode[1], s=100, c="r", marker="s", label="Root Node"
    )
    ax.scatter(
        goalnode[0], goalnode[1], s=100, c="g", marker="^", label="Goal Node"
    )
    for i, neighbors in graph.items():
        for j, _ in neighbors:
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                "k--",
                alpha=0.2,
            )
    ax.plot(qpath[:, 0], qpath[:, 1], "r-o", linewidth=2, label="Shortest Path")
    ax.set_title("Random Geometric Graph with Dijkstra's Paths")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # method1()
    # method2()
    method3()
    # allnode_RGG()
