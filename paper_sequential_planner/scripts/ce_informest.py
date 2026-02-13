import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from geometric_ellipse import *

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))

collision_points = dataset[dataset[:, 2] > 0]
free_points = dataset[dataset[:, 2] < 0]


collision_kdt = KDTree(collision_points)
free_kdt = KDTree(free_points)

# q = np.array([0.0, 0.0]).reshape(1, -1)
# d, i = kdt.query(q, k=5)
# print("Query point:", q)
# print("Nearest neighbors' indices:", i)
# print("Nearest neighbors' distances:", d)


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2], cmap="bwr", s=5)
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("C-space dataset")
ax.set_aspect("equal")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
plt.show()
