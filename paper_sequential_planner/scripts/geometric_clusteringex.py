import numpy as np
import matplotlib.pyplot as plt
from geometric_Xmean import fit
import os

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


cspace_dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
cspace_nearest = np.load(os.path.join(rsrc, "cspace_dataset_nearest_distance.npy"))


def view_dataset():
    vmax = np.max(cspace_nearest[:, 2])
    vmin = np.min(cspace_nearest[:, 2])
    d = cspace_nearest[:, 2]

    fig, ax = plt.subplots()
    sc = ax.scatter(
        cspace_nearest[:, 0],
        cspace_nearest[:, 1],
        c=d,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.plot(
        cspace_obs[:, 0],
        cspace_obs[:, 1],
        "ro",
        markersize=2,
        label="Obstacle C-space",
        alpha=0.3,
    )
    plt.colorbar(sc, label="Nearest Distance to Obstacle")
    ax.set_xlabel("Theta 1")
    ax.set_ylabel("Theta 2")
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def Xmeans_clustering():
    dof = 2
    kmin = 3
    kmax = 40
    weights = [0.5] * dof

    xmeans = fit(cspace_obs[:, 0:dof], kmin=kmin, kmax=kmax, weights=weights)
    N = xmeans.k
    labels = xmeans.labels_
    center = xmeans.centroid_centres_
    points_per_cluster = xmeans.count

    print("Number of clusters assigned: %d." % N)
    print("Cluster centers:\n", center)
    print("labels:\n", labels)
    print("Points per cluster:\n", points_per_cluster)

    group1_point = cspace_obs[labels == 0]
    group2_point = cspace_obs[labels == 1]
    group3_point = cspace_obs[labels == 2]

    fig, ax = plt.subplots()
    ax.plot(
        center[:, 0],
        center[:, 1],
        "kx",
        markersize=10,
        markeredgewidth=2,
        label="Centroids",
    )
    ax.plot(
        group1_point[:, 0],
        group1_point[:, 1],
        "ro",
        markersize=2,
        label="Cluster 1",
    )
    ax.plot(
        group2_point[:, 0],
        group2_point[:, 1],
        "go",
        markersize=2,
        label="Cluster 2",
    )
    ax.plot(
        group3_point[:, 0],
        group3_point[:, 1],
        "bo",
        markersize=2,
        label="Cluster 3",
    )
    ax.set_xlabel("Theta 1")
    ax.set_ylabel("Theta 2")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    view_dataset()
    Xmeans_clustering()
