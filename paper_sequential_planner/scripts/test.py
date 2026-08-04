# sampling technique to do node sparsification
# all these sampling techniques is a bit expensive
# Poisson disk / blue-noise sampling
# Farthest Point Sampling (FPS) / Max-min sampling / downsample
# Adaptive / density-aware sparsification
# Incremental sparsification (online PRM)
from matplotlib.collections import PatchCollection
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os
import fpsample
from paper_sequential_planner.scripts.geometric_ellipse import *
from paper_sequential_planner.scripts.rtsp_lazyprm import *

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))


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


def point_knn_sparse():
    points = np.random.rand(10, 2)
    new_points = np.random.rand(20, 2)
    eps = 0.05
    k = 6
    delta = 0.1

    tree = KDTree(points)
    pn = tree.query_ball_point(new_points, eps)
    keepbool = np.array([len(neighbors) == 0 for neighbors in pn])
    pkeeps = new_points[keepbool]

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=50, label="Existing")
    ax.scatter(new_points[:, 0], new_points[:, 1], s=50, label="New")
    ax.scatter(
        pkeeps[:, 0],
        pkeeps[:, 1],
        s=100,
        facecolors="none",
        edgecolors="r",
        label="Kept",
    )
    ax.set_title("Before filtering")
    ax.set_aspect("equal")
    ax.legend()
    plt.grid()
    plt.show()


def poisson_disk2():
    # Poisson disk sampling
    # r = 0.05 -> shape (253, 2) filled in unit square
    radius = 0.05
    engine = qmc.PoissonDisk(d=2, radius=radius)
    Xrand = engine.random(500)
    print(f"==>> Xrand.shape: \n{Xrand.shape}")

    # scale to [-pi, pi]
    Xrand = Xrand * 2 * np.pi - np.pi  # scale to [-pi, pi]
    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)

    Xranddist = np.linalg.norm(Xrand, axis=1)
    incircle = Xranddist <= np.pi
    Xrandball = Xrand[incircle]

    xstart = np.array([0.15, 0.60]).reshape(2, 1)
    xgoal = np.array([2.0, 1.8]).reshape(2, 1)

    cMin = np.linalg.norm(xgoal - xstart)
    cMax = 1.5 * cMin
    Xinf = informed_sampling_bulk(xstart, xgoal, cMax, numsample=500)
    xCenter, rotationAxisC, L, cMin = informed_sampling_ellipse(
        xstart, xgoal, cMax
    )
    states = isPointinEllipseBulk2(xCenter, rotationAxisC, L, Xrand)
    XrandinEllipse = Xrand[states]

    ax.plot(xstart[0], xstart[1], "go", markersize=8, label="Start")
    ax.plot(xgoal[0], xgoal[1], "bo", markersize=8, label="Goal")

    ax.scatter(Xrand[:, 0], Xrand[:, 1])
    ax.scatter(Xinf[:, 0], Xinf[:, 1], color="m", s=20, label="Informed")

    rcir = radius * 2 * np.pi / 2
    circles = [plt.Circle((xi, yi), radius=rcir, fill=False) for xi, yi in Xrand]
    collection = PatchCollection(circles, match_original=True)
    ax.add_collection(collection)

    cir = plt.Circle((0, 0), radius=np.pi, fill=False, color="g")
    ax.add_patch(cir)

    ax.scatter(
        XrandinEllipse[:, 0],
        XrandinEllipse[:, 1],
        color="r",
        s=20,
        label="In Ellipse",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
    plt.show()


def poisson_disk3():
    # Poisson disk sampling 3d
    # r = 0.05 -> shape (4713, 3) filled in unit cube
    radius = 0.05
    engine = qmc.PoissonDisk(d=3, radius=radius)
    Xrand3d = engine.random(5000)
    Xrand3d = Xrand3d * 2 * 0.5 - 0.5  # scale to [-0.5, 0.5]
    print(f"==>> Xrand3d.shape: \n{Xrand3d.shape}")
    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=0.5)
    box = trimesh.creation.box(extents=(1, 1, 1))
    box.visual.face_colors = [100, 150, 255, 40]
    scene.add_geometry(box)
    scene.add_geometry(axis)
    pc = trimesh.points.PointCloud(Xrand3d, colors=[255, 0, 0, 255])
    scene.add_geometry(pc)
    scene.show()


def poisson_disk_highdim():
    # r = 0.05 -> shape (93942, 4) filled in unit hypercube
    radius = 0.05
    engine = qmc.PoissonDisk(d=4, radius=radius)
    Xrand4d = engine.random(100000)
    Xrand4d = Xrand4d * 2 * 0.5 - 0.5  # scale to [-0.5, 0.5]
    print(f"==>> Xrand4d.shape: \n{Xrand4d.shape}")

    engine = qmc.PoissonDisk(d=5, radius=radius)
    Xrand5d = engine.random(3000000)
    Xrand5d = Xrand5d * 2 * 0.5 - 0.5  # scale to [-0.5, 0.5]
    print(f"==>> Xrand5d.shape: \n{Xrand5d.shape}")

    # Poisson disk sampling 6d
    # Unable to allocate 309. GiB for an array with shape (49, 49, 49, 49, 49, 49, 6) and data type float32
    engine = qmc.PoissonDisk(d=6, radius=radius)
    Xrand6d = engine.random(700000000)
    Xrand6d = Xrand6d * 2 * 0.5 - 0.5  # scale to [-0.5, 0.5]
    print(f"==>> Xrand6d.shape: \n{Xrand6d.shape}")


def fpsample_2test():
    pc = np.random.rand(4096, 3) * 2 - 1  # scale to [-1, 1]
    kdtree_fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, 1024)
    pc_fps = pc[kdtree_fps_samples_idx]
    pc_sparse = sparsify_nodes(pc_fps, eps=0.2)

    print(f"==>> pc.shape: \n{pc.shape}")
    print(f"==>> pc_fps.shape: \n{pc_fps.shape}")
    print(f"==>> pc_sparse.shape: \n{pc_sparse.shape}")

    fig, ax = plt.subplots()
    ax.plot(pc[:, 0], pc[:, 1], "ro", markersize=3, label="Original")
    ax.plot(pc_fps[:, 0], pc_fps[:, 1], "go", markersize=3, label="FPS")
    ax.plot(pc_sparse[:, 0], pc_sparse[:, 1], "bo", markersize=3, label="Sparse")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    plt.show()


def fpsample_3test():
    pc = np.random.rand(4096, 3) * 2 - 1  # scale to [-1, 1]
    kdtree_fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, 1024)
    pc_fps = pc[kdtree_fps_samples_idx]
    pc_sparse = sparsify_nodes(pc_fps, eps=0.2)

    print(f"==>> pc.shape: \n{pc.shape}")
    print(f"==>> pc_fps.shape: \n{pc_fps.shape}")
    print(f"==>> pc_sparse.shape: \n{pc_sparse.shape}")

    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, 1024, h=3)
    pc_fps = pc[kdline_fps_samples_idx]

    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=0.5)
    box = trimesh.creation.box(extents=(1, 1, 1))
    box.visual.face_colors = [100, 150, 255, 40]
    scene.add_geometry(box)
    scene.add_geometry(axis)

    pc = trimesh.points.PointCloud(pc, colors=[255, 0, 0, 255])
    scene.add_geometry(pc)
    pc_fps = trimesh.points.PointCloud(pc_fps, colors=[0, 255, 0, 255])
    scene.add_geometry(pc_fps)
    scene.show()


def fpsample_highdim():
    pc = np.random.rand(600000000, 6) * 2 - 1  # scale to [-1, 1]
    kdtree_fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, 600000000)
    pc_fps = pc[kdtree_fps_samples_idx]
    print(f"==>> pc.shape: \n{pc.shape}")
    print(f"==>> pc_fps.shape: \n{pc_fps.shape}")


class SceneOMPLPlannerGoalset:

    def __init__(self, *args, **kwargs):
        from ompl import base as ob
        from ompl import geometric as og
        from ompl import util as ou

        self.ob = ob
        self.og = og
        self.ou = ou

        ou.RNG.setSeed(kwargs.get("seed", 42))
        self._lm = np.array(
            [
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-np.pi, np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
            ]
        )
        self.timeout = kwargs.get("timeout", 10.0)
        self.limits = kwargs.get("limits", self._lm)

        self.space = ob.RealVectorStateSpace(6)
        self.bounds = ob.RealVectorBounds(6)
        for i in range(6):
            self.bounds.setLow(i, self.limits[i, 0])
            self.bounds.setHigh(i, self.limits[i, 1])
        self.space.setBounds(self.bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )
        self._p = kwargs.get("planner", "BITstar")
        if self._p == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif self._p == "ABITstar":
            self.planner = og.ABITstar(self.ss.getSpaceInformation())
        elif self._p == "AITstar":
            self.planner = og.AITstar(self.ss.getSpaceInformation())

        self._r = kwargs.get("range", None)
        if self._r is not None:
            self.planner.setRange(self._r)

        obj = ob.PathLengthOptimizationObjective(self.ss.getSpaceInformation())
        self.ss.setOptimizationObjective(obj)

        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        q = [state[0], state[1], state[2], state[3], state[4], state[5]]
        return True

    def query_planning(self):
        # Important!
        # Clear previous planning data to ensure fresh planning because caching
        self.ss.clear()

        start_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # goal_list = np.array(
        #     [
        #         [1.57, -1.57, 1.3, 1.3, 0.5, 1.57],
        #         [-1.57, -1.57, 1.3, 1.3, -1.2, -1.57],
        #         [1.57, 1.57, -1.3, -0.4, 1.2, 1.57],
        #         [-1.57, 1.57, -1.3, -1.3, -0.5, -1.57],
        #     ]
        # )
        goal_list = np.random.uniform(
            low=self.limits[:, 0], high=self.limits[:, 1], size=(256, 6)
        )

        start = self.ob.State(self.space)
        start[0] = start_list[0]
        start[1] = start_list[1]
        start[2] = start_list[2]
        start[3] = start_list[3]
        start[4] = start_list[4]
        start[5] = start_list[5]
        self.ss.setStartState(start)

        goal = self.ob.GoalStates(self.ss.getSpaceInformation())
        for i in range(goal_list.shape[0]):
            goal_state = self.ob.State(self.space)
            goal_state[0] = goal_list[i, 0]
            goal_state[1] = goal_list[i, 1]
            goal_state[2] = goal_list[i, 2]
            goal_state[3] = goal_list[i, 3]
            goal_state[4] = goal_list[i, 4]
            goal_state[5] = goal_list[i, 5]
            goal.addState(goal_state)
        self.ss.setGoal(goal)

        dist = np.linalg.norm(goal_list - start_list, axis=1)

        status = self.ss.solve(10.0)
        print("Plan from ", start_list, " to ", goal_list, "estimate cost:", dist)
        (
            print("EXACT")
            if status.getStatus() == status.EXACT_SOLUTION
            else print("Invalid result")
        )
        if status.getStatus() == status.EXACT_SOLUTION:
            self.ss.simplifySolution()
            path = self.ss.getSolutionPath()
            path_cost = path.length()

            print("Found solution:")
            print(f"Path cost: {path_cost}")
            print(self.ss.getSolutionPath())

            pathlist = []
            for i in range(path.getStateCount()):
                pi = path.getState(i)
                pathlist.append([pi[0], pi[1], pi[2], pi[3], pi[4], pi[5]])

            i = None
            for j, g in enumerate(goal_list):
                if np.allclose(pathlist[-1], g, atol=1e-6):
                    i = j
                    break
            goalid = i

            return pathlist, path_cost, goalid
        else:
            print("No solution found")
            return None


if __name__ == "__main__":
    # point_knn_sparse()
    # poisson_disk2()
    # poisson_disk3()
    # poisson_disk_highdim()
    # fpsample_2test()
    # fpsample_3test()
    # fpsample_highdim()
    p = SceneOMPLPlannerGoalset()
    path, cost, gi = p.query_planning()
    print(f"==>> path: \n{path}")
    print(f"==>> cost: \n{cost}")
    print(f"==>> gi: \n{gi}")
