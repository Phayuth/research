# sampling technique to do node sparsification
# Poisson disk / blue-noise sampling
# Farthest Point Sampling (FPS) / Max-min sampling / downsample
# Adaptive / density-aware sparsification
# Incremental sparsification (online PRM)
from matplotlib.collections import PatchCollection
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from paper_sequential_planner.scripts.rtsp_lazyprmsparse import *

np.random.seed(42)

X = np.random.uniform(-np.pi, np.pi, size=(500, 2))
Xs = bulk_collisioncheck(X)
points = X[Xs == 0]  # only keep collision-free nodes
points_sparse = sparsify_nodes(points, eps=0.05 * 2 * np.pi)  # node sparse

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
    points_sparse[:, 0],
    points_sparse[:, 1],
    s=50,
    c="b",
    label="Sparse Nodes",
)
for i, node in enumerate(points_sparse):
    ax.text(node[0], node[1], str(i), fontsize=8, color="red", ha="right")
ax.set_aspect("equal")
ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid()
plt.show()


# Poisson disk sampling
radius = 0.05
engine = qmc.PoissonDisk(d=2, radius=radius)
Xrand = engine.random(500)
print(f"==>> Xrand.shape: \n{Xrand.shape}")

# scale to [-pi, pi]
Xrand = Xrand * 2 * np.pi - np.pi  # scale to [-pi, pi]
fig, ax = plt.subplots()
ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
_ = ax.scatter(Xrand[:, 0], Xrand[:, 1])
circles = [
    plt.Circle((xi, yi), radius=radius * 2 * np.pi / 2, fill=False)
    for xi, yi in Xrand
]
collection = PatchCollection(circles, match_original=True)
ax.add_collection(collection)
ax.set_aspect("equal")
ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
plt.show()

Xrandsparse = sparsify_nodes(Xrand, eps=0.05 * 2 * np.pi)  # node sparse
fig, ax = plt.subplots()
ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
ax.scatter(
    Xrand[:, 0],
    Xrand[:, 1],
    s=100,
    c="lightgray",
    marker="x",
    label="Original Nodes",
)
ax.scatter(
    Xrandsparse[:, 0],
    Xrandsparse[:, 1],
    s=50,
    c="b",
    label="Sparse Nodes",
)
for i, node in enumerate(Xrandsparse):
    ax.text(node[0], node[1], str(i), fontsize=8, color="red", ha="right")
ax.set_aspect("equal")
ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid()
plt.show()


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
