from sklearn.cluster import SpectralClustering, DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


# cspace_dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
# cspace_nearest = np.load(os.path.join(rsrc, "cspace_dataset_nearest_distance.npy"))

# npoints = 2000
# print(cspace_obs.shape)
# samples_id = np.random.choice(
#     range(cspace_obs.shape[0]), size=npoints, replace=False
# )
dataset = cspace_obs
# dataset = cspace_obs[samples_id]

# dof = 2
# kmin = 3
# kmax = 40

# data1 = np.random.multivariate_normal(
#     mean=[1.0] * dof,
#     cov=np.diag([0.5] * dof),
#     size=500,
# )
# data2 = np.random.multivariate_normal(
#     mean=[5.0] * dof,
#     cov=np.diag([0.8] * dof),
#     size=500,
# )
# data3 = np.random.multivariate_normal(
#     mean=[8.0] * dof,
#     cov=np.diag([0.3] * dof),
#     size=500,
# )
# dataset = np.vstack((data1, data2, data3))
# print(dataset.shape)


# clustering = SpectralClustering(
#     n_clusters=4,
#     eigen_solver="arpack",
#     affinity="nearest_neighbors",
#     random_state=42,
# )

clustering = DBSCAN(eps=0.5, min_samples=5)
clustering.fit(dataset)
labels = clustering.labels_

fig, ax = plt.subplots()
sc = ax.scatter(
    dataset[:, 0],
    dataset[:, 1],
    c=labels,
    cmap="viridis",
)
ax.set_title("DBSCAN Clustering of C-space Obstacles")
ax.set_aspect("equal", "box")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.grid(True)
plt.show()
