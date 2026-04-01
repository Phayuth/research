import numpy as np

# here we want to develop a cluster of task to reduce num edges
# cluster cost metric is to be discussed later
# we want to identify task clusters that are close, meaning it learn
# they belong to the same topological region
# also we want to identify task cluster to cluster as well
# but moving from task to task is not straightforward, we need to consider the cspace connectivity as well


# Combined SE(3) metric (common practice) -----------------------------
def rotation_angle(R1, R2):
    R_err = R1.T @ R2
    tr = np.trace(R_err)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  # radians


def se3_error_split(T1, T2):
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    et = np.linalg.norm(t1 - t2)
    er = rotation_angle(R1, R2)
    return et, er


def se3_error_weighted(T1, T2, w_rot=1.0):
    # e=sqrt ∥t1​−t2​∥2+λθ2
    et, er = se3_error_split(T1, T2)
    return np.sqrt(et**2 + (w_rot * er) ** 2)


# Alternative (Lie algebra, cleaner for optimization) -----------------------------
def so3_log(R):
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])


def se3_log(T):
    R = T[:3, :3]
    t = T[:3, 3]

    phi = so3_log(R)
    theta = np.linalg.norm(phi)

    if theta < 1e-8:
        V_inv = np.eye(3)
    else:
        axis = phi / theta
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta**2)
        V = np.eye(3) + B * K + ((1 - A) / theta**2) * (K @ K)
        V_inv = np.linalg.inv(V)

    rho = V_inv @ t
    return np.hstack([rho, phi])  # 6D vector


def se3_error_log(T1, T2):
    T_err = np.linalg.inv(T1) @ T2
    xi = se3_log(T_err)
    return xi, np.linalg.norm(xi)


def quat_angle(q1, q2):
    dot = np.abs(np.dot(q1, q2))  # double-cover fix
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pytransform3d.transformations import plot_transform
    from pytransform3d.plot_utils import make_3d_axis

    y = np.linspace(-0.5, 0.5, 5)
    z = np.linspace(-0.5, 0.5, 5)
    Y, Z = np.meshgrid(y, z)
    YZ = np.vstack([Y.ravel(), Z.ravel()]).T
    x = np.full(YZ.shape[0], 0.42)
    points1 = np.hstack([x[:, None], YZ])

    points2 = points1.copy()
    points2[:, 0] = -0.42

    points = np.vstack([points1, points2])
    n = points.shape[0]

    Hlist = np.empty((n, 4, 4))
    for i, p in enumerate(points):
        H = np.eye(4)
        H[0:3, 3] = p
        Hlist[i] = H

    # H1 = np.eye(4)
    # H1[0:3, 3] = [1.0, 0.0, 0.0]
    # H2 = np.eye(4)
    # H2[0:3, 3] = [1.0, 0.0, 0.1]
    # print("Split error:", se3_error_split(H1, H2))
    # print("Weighted error:", se3_error_weighted(H1, H2, w_rot=10.0))
    # print("Log error:", se3_error_log(H1, H2))

    X = np.array([se3_log(H) for H in Hlist])  # shape (N,6)
    print(f"==>> X: \n{X}")
    # Normalize so translation/rotation are comparable:
    X[:, :3] /= np.std(X[:, :3]) + 1e-8
    X[:, 3:] /= np.std(X[:, 3:]) + 1e-8
    print(f"==>> X: \n{X}")

    # option 1
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    def estimate_eps(X, k=10):
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        kth = np.sort(dists[:, -1])
        return np.percentile(kth, 90)  # automatic-ish

    eps = estimate_eps(X)
    print(f"==>> eps: \n{eps}")
    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
    print(f"==>> labels: \n{labels}")
    cluster_id = np.unique(labels)
    print(f"==>> cluster_id: \n{cluster_id}")
    num_clusters = len(cluster_id[cluster_id != -1])  # ignore noise
    print(f"==>> num_clusters: \n{num_clusters}")

    # option 2
    # import hdbscan

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    # labels = clusterer.fit_predict(X)

    # option 3
    from sklearn.mixture import GaussianMixture

    def fit_gmm_bic(X, k_max=10):
        best_k, best_model, best_bic = None, None, np.inf
        for k in range(1, k_max + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full")
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_k, best_model, best_bic = k, gmm, bic
        return best_k, best_model

    k, gmm = fit_gmm_bic(X)
    labels_gmm = gmm.predict(X)
    print(f"==>> labels_gmm: \n{labels_gmm}")

    # find mean H for each cluster
    def se3_mean(Ts, max_iter=20):
        T_mean = Ts[0].copy()

        for _ in range(max_iter):
            xis = [se3_log(np.linalg.inv(T_mean) @ T) for T in Ts]
            xi_bar = np.mean(xis, axis=0)

            if np.linalg.norm(xi_bar) < 1e-6:
                break

            # exponential map (approx)
            rho, phi = xi_bar[:3], xi_bar[3:]
            theta = np.linalg.norm(phi)

            if theta < 1e-8:
                R = np.eye(3)
            else:
                axis = phi / theta
                K = np.array(
                    [
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ]
                )
                R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

            T_update = np.eye(4)
            T_update[:3, :3] = R
            T_update[:3, 3] = rho

            T_mean = T_mean @ T_update

        return T_mean

    Hmean1 = se3_mean(Hlist[labels == 0])
    Hmean2 = se3_mean(Hlist[labels == 1])
    print(f"==>> Hmean1: \n{Hmean1}")
    print(f"==>> Hmean2: \n{Hmean2}")

    nbrs = NearestNeighbors(n_neighbors=10).fit(X)
    dists, indices = nbrs.kneighbors(X)
    print(f"==>> dists: \n{dists}")
    print(f"==>> indices: \n{indices}")

    X0 = X[0]
    X0neigh = X[indices[0]]
    print(f"==>> X0: \n{X0}")
    print(f"==>> X0neigh: \n{X0neigh}")
    H0 = Hlist[0]
    H0neigh = Hlist[indices[0]]
    print(f"==>> H0: \n{H0}")
    print(f"==>> H0neigh: \n{H0neigh}")

    ax = make_3d_axis(1)
    plot_transform(ax, name="world")
    plot_transform(ax, Hmean1, s=0.1, name="Hmean1")
    plot_transform(ax, Hmean2, s=0.1, name="Hmean2")
    for i, H in enumerate(Hlist):
        plot_transform(ax, H, s=0.05, name=f"Cluster {labels[i]}")
    for hn in H0neigh:
        ax.plot(
            [H0[0, 3], hn[0, 3]],
            [H0[1, 3], hn[1, 3]],
            [H0[2, 3], hn[2, 3]],
            "k--",
            alpha=0.5,
        )
    ax.set_title("SE(3) Error Metrics")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.show()
