# sampling technique to do node sparsification
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
from paper_sequential_planner.scripts.geometric_ellipse import *

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))


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


poisson_disk2()


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


def poisson_disk6():
    # Poisson disk sampling 6d
    # Unable to allocate 309. GiB for an array with shape (49, 49, 49, 49, 49, 49, 6) and data type float32
    radius = 0.05
    engine = qmc.PoissonDisk(d=6, radius=radius)
    Xrand6d = engine.random(5000)
    Xrand6d = Xrand6d * 2 * 0.5 - 0.5  # scale to [-0.5, 0.5]
    print(f"==>> Xrand6d.shape: \n{Xrand6d.shape}")


#
# kdt = KDTree(points)
# dists, indices = kdt.query(points_sparse, distance_upper_bound=0.05 * 2 * np.pi)
# print(f"==>> dists: \n{dists}")
# print(f"==>> indices: \n{indices}")


# points = np.random.rand(10, 2)
# new_points = np.random.rand(20, 2)
# eps = 0.05
# k = 6
# delta = 0.1

# tree = KDTree(points)
# pn = tree.query_ball_point(new_points, eps)
# keepbool = np.array([len(neighbors) == 0 for neighbors in pn])
# pkeeps = new_points[keepbool]

# fig, ax = plt.subplots()
# ax.scatter(points[:, 0], points[:, 1], s=50, label="Existing")
# ax.scatter(new_points[:, 0], new_points[:, 1], s=50, label="New")
# ax.scatter(
#     pkeeps[:, 0],
#     pkeeps[:, 1],
#     s=100,
#     facecolors="none",
#     edgecolors="r",
#     label="Kept",
# )
# ax.set_title("Before filtering")
# ax.set_aspect("equal")
# ax.legend()
# plt.grid()
# plt.show()
