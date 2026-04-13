from ctypes import util
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

# here we want to develop a cluster of task to reduce num edges
# cluster cost metric is to be discussed later
# we want to identify task clusters that are close, meaning it learn
# they belong to the same topological region
# also we want to identify task cluster to cluster as well
# but moving from task to task is not straightforward, we need to consider the cspace connectivity as well
# if we fail to identify the cluster, we will have to connect all task to all task, which is O(n^2) edges, which is not scalable
# if cluster only found 1: meaning it is either the same topology or they dont correlate at all, we need to decide on that


# Combined SE(3) metric (common practice) -----------------------------
def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)


def rotation_error(R1, R2):
    R_err = R1.T @ R2
    tr = np.trace(R_err)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  # radians


def rotation_quat_error(q1, q2):
    """
    >>> q1 = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90 deg around X
    >>> q2 = np.array([0.7071, 0.0, 0.7071, 0.0])  # 90 deg around Y
    >>> print("Quat angle:", rotation_quat_error(q1, q2))

    >>> R1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # 90 deg around X
    >>> R2 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # 90 deg around Y
    >>> print("SO(3) log:", rotation_error(R1, R2))
    """
    dot = np.abs(np.dot(q1, q2))  # double-cover fix
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot)


def se3_error_split(H1, H2):
    R1, t1 = H1[:3, :3], H1[:3, 3]
    R2, t2 = H2[:3, :3], H2[:3, 3]

    et = translation_error(t1, t2)
    er = rotation_error(R1, R2)
    return et, er


def se3_error(H1, H2, w_rot=1.0):
    # e=sqrt ∥t1​−t2​∥2+λθ2
    et, er = se3_error_split(H1, H2)
    return np.sqrt(et**2 + (w_rot * er) ** 2)


def se3_error_pairwise_distance(H, w_rot=1.0):
    """
    H: (N, 4, 4)
    return: (N, N) pairwise SE(3) distance

    Eqivalent to:
    Dist = np.zeros((len(Hlist), len(Hlist)))
    for i in range(len(Hlist)):
        for j in range(i + 1, len(Hlist)):
            Dist[i, j] = se3_error(Hlist[i], Hlist[j], w_rot=10.0)
            Dist[j, i] = Dist[i, j]
    """
    R = H[:, :3, :3]  # (N, 3, 3)
    t = H[:, :3, 3]  # (N, 3)

    dt = t[:, None, :] - t[None, :, :]  # (N, N, 3)
    et2 = np.sum(dt**2, axis=-1)  # (N, N)

    # (N, N, 3, 3)
    R_rel = np.matmul(R.transpose(0, 2, 1)[:, None, :, :], R[None, :, :, :])

    tr = np.trace(R_rel, axis1=-2, axis2=-1)  # (N, N)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (N, N)

    d = np.sqrt(et2 + (w_rot * theta) ** 2)
    return d


def se3_error_position_only_pairwise_distance(H):
    t = H[:, :3, 3]  # (N, 3)
    dt = t[:, None, :] - t[None, :, :]  # (N, N, 3)
    et2 = np.sum(dt**2, axis=-1)  # (N, N)
    d = np.sqrt(et2)
    return d


# Alternative (Lie algebra, cleaner for optimization) -----------------------------
def so3_log(R):
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])


def se3_log(H):
    R = H[:3, :3]
    t = H[:3, 3]

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


def se3_error_log(H1, H2):
    H_err = np.linalg.inv(H1) @ H2
    xi = se3_log(H_err)
    return xi, np.linalg.norm(xi)


# cluster algorithm ----------------------------------------------------------
def dbscan_clustering(Hse3logerr):

    def _estimate_eps(X, k=10):
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        kth = np.sort(dists[:, -1])
        return np.percentile(kth, 90)  # automatic-ish

    eps = _estimate_eps(Hse3logerr)
    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(Hse3logerr)
    cluster_id = np.unique(labels)
    num_clusters = len(cluster_id[cluster_id != -1])  # ignore noise
    return labels, cluster_id, num_clusters


def dhbscan_clustering(Hse3logerr):
    # import hdbscan
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    # labels = clusterer.fit_predict(X)
    pass


def gmm_bic_clustering(Hse3logerr):

    def _fit_gmm_bic(X, k_max=10):
        best_k, best_model, best_bic = None, None, np.inf
        for k in range(1, k_max + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full")
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_k, best_model, best_bic = k, gmm, bic
        return best_k, best_model

    best_k, best_gmm = _fit_gmm_bic(Hse3logerr)
    labels = best_gmm.predict(Hse3logerr)
    cluster_id = np.unique(labels)
    num_clusters = len(cluster_id[cluster_id != -1])  # ignore noise
    return labels, cluster_id, num_clusters


# find mean of SE(3) poses in a cluster ----------------------------------
def se3_mean(Hs, max_iter=20):
    H_mean = Hs[0].copy()

    for _ in range(max_iter):
        xis = [se3_log(np.linalg.inv(H_mean) @ H) for H in Hs]
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

        H_mean = H_mean @ T_update

    return H_mean


# pose format conversion ------------------------------------------------------
# X = (x,y,z, qx, qy, qz, qw) # shape (7,)
# Xlist = [(x,y,z, qx, qy, qz, qw), ...] # shape (N,7)
# H = [R|t] in SE(3) # shape (4,4)
# Hlist = [H1, H2, ...] # shape (N,4,4)


# testing
# Hlist = poses_a()
# Xlist = Hlist_to_Xlist(Hlist)
# Hlistrecover = Xlist_to_Hlist(Xlist)
# t = np.allclose(Hlist, Hlistrecover, atol=1e-6)
def H_to_X(H):
    t = H[:3, 3]
    R_mat = H[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()
    X = np.hstack([t, quat])
    return X


def Xlist_to_Hlist(Xlist):
    Hlist = []
    for X in Xlist:
        t = X[:3]
        quat = X[3:]
        R_mat = R.from_quat(quat).as_matrix()
        H = np.eye(4)
        H[:3, :3] = R_mat
        H[:3, 3] = t
        Hlist.append(H)
    return np.array(Hlist)


def Hlist_to_Xlist(Hlist):
    Xlist = []
    for H in Hlist:
        t = H[:3, 3]
        R_mat = H[:3, :3]
        quat = R.from_matrix(R_mat).as_quat()
        X = np.hstack([t, quat])
        Xlist.append(X)
    return np.array(Xlist)


# Repair pose format -----------------------------------------------------------
# due to some robot dofs is not 6dof, we concat the pose with some dummy values
# X = (x,y,z, qx, qy, qz, qw) # shape (7,)
# H = [R|t] in SE(3) # shape (4,4)
def xlist_to_Xlist(xlist):
    if xlist.shape[1] == 2:  # 2DOF robot - (ntasks, 2), z=0, quat=(0,0,0,1)
        ntasks = xlist.shape[0]
        dummy = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]] * ntasks)
        Xlist = np.column_stack([xlist, dummy])
        return Xlist
    elif xlist.shape[1] == 3:  # 3DOF robot - (ntasks, 3), quat=(0,0,0,1)
        ntasks = xlist.shape[0]
        dummy = np.array([[0.0, 0.0, 0.0, 1.0]] * ntasks)
        Xlist = np.column_stack([xlist, dummy])
        return Xlist


# generate synthetic poses ----------------------------------------------
def poses_a():
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
    return Hlist


def poses_b():
    # add a bit of noise on rotation now cluster is not perfect, but we can still find the cluster
    def _gen_linear_H(s, e, quat, num_tasks=10):
        t = np.linspace(s, e, num_tasks)
        Hlist = [np.eye(4) for _ in range(num_tasks)]
        for i in range(num_tasks):
            Hlist[i][:3, 3] = t[i]
            Hlist[i][:3, :3] = R.from_quat(quat).as_matrix()
        return Hlist

    def _Hrot_Z(a):
        H = np.eye(4)
        c, s = np.cos(a), np.sin(a)
        H[0:3, 0:3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        return H

    def _RotPI(H):
        Hdh_to_urdf = _Hrot_Z(np.pi)
        return np.linalg.inv(Hdh_to_urdf) @ H

    def _RotPI2(H):
        Hdh_to_urdf = _Hrot_Z(np.pi / 2)
        return np.linalg.inv(Hdh_to_urdf) @ H

    def _RotPI3(H):
        Hdh_to_urdf = _Hrot_Z(3 * np.pi / 2)
        return np.linalg.inv(Hdh_to_urdf) @ H

    size = 4
    params = {
        0: ([0.5, 0.5, 0.6], [0.5, -0.5, 0.6], [0.0, 0.707106, 0.0, 0.707106]),
        1: ([0.5, 0.5, 0.4], [0.5, -0.5, 0.4], [0.0, 0.707106, 0.0, 0.707106]),
        2: ([0.5, 0.5, 0.2], [0.5, -0.5, 0.2], [0.0, 0.707106, 0.0, 0.707106]),
        3: ([0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.707106, 0.0, 0.707106]),
    }
    HH = []
    for k in params:
        s, e, quat = params[k]
        quat_noise = quat + np.random.normal(0, 0.05, size=4)
        HH += _gen_linear_H(s, e, quat_noise, num_tasks=size)
    GG = []
    for h in HH:
        GG.append(_RotPI(h))
        GG.append(_RotPI2(h))
        GG.append(_RotPI3(h))
    Hlist = np.array(HH + GG)
    return Hlist


def poses_c():
    size = 100
    Hlist = np.empty((size, 4, 4))
    for i in range(size):
        t = np.random.uniform(-1, 1, size=(3,))
        quat = np.random.uniform(-np.pi, np.pi, size=(4,))
        RR = R.from_quat(quat)
        H = np.eye(4)
        H[:3, :3] = RR.as_matrix()
        H[:3, 3] = t
        Hlist[i] = H
    return Hlist


def poses_d():
    def _gen_linear_H(s, e, quat, num_tasks=10):
        t = np.linspace(s, e, num_tasks)
        Hlist = [np.eye(4) for _ in range(num_tasks)]
        for i in range(num_tasks):
            Hlist[i][:3, 3] = t[i]
            Hlist[i][:3, :3] = R.from_quat(quat).as_matrix()
        return Hlist

    size = 4
    params = {
        0: ([-0.4, 0.6, 0.5], [0.4, 0.6, 0.5], [-0.707106, 0.0, 0.0, 0.707106]),
        1: ([-0.4, 0.6, 0.2], [0.4, 0.6, 0.2], [-0.707106, 0.0, 0.0, 0.707106]),
        2: ([-0.6, -0.4, 0.5], [-0.6, 0.4, 0.5], [-0.5, -0.5, 0.5, 0.5]),
        3: ([-0.6, -0.4, 0.2], [-0.6, 0.4, 0.2], [-0.5, -0.5, 0.5, 0.5]),
        4: ([0.4, -0.6, 0.5], [-0.4, -0.6, 0.5], [0.0, -0.707106, 0.707106, 0.0]),
        5: ([0.4, -0.6, 0.2], [-0.4, -0.6, 0.2], [0.0, -0.707106, 0.707106, 0.0]),
    }
    HH = []
    for k in params:
        s, e, quat = params[k]
        quat_noise = quat + np.random.normal(0, 0.05, size=4)
        HH += _gen_linear_H(s, e, quat_noise, num_tasks=size)
    Hlist = np.array(HH)
    return Hlist


def transform_lookat(at, eye, up):
    """Copied from OpenRAVE's transformLookat function in "geometry.h".

    Returns an end effector transform matrix that looks along a ray with a desired up vector (corresponding to y axis of the end effector).
    If up vector is parallel to ray, tries to use +y or +x direction instead.
    If ray length is zero, chooses ray to be +z direction by default.

    @param at the point space to look at, the camera will rotation and zoom around this point
    @param eye the position of the camera in space
    @param up desired end effector y axis direction
    @return end effector transform matrix
    """
    vdir = np.array(at) - eye
    if np.linalg.norm(vdir) > 1e-6:
        vdir *= 1 / np.linalg.norm(vdir)
    else:
        vdir = [0.0, 0.0, 1.0]

    vup = np.array(up) - vdir * np.dot(up, vdir)
    if np.linalg.norm(vup) < 1e-8:
        vup = [0.0, 1.0, 0.0]
        vup -= vdir * np.dot(vdir, vup)
        if np.linalg.norm(vup) < 1e-8:
            vup = [1.0, 0.0, 0.0]
            vup -= vdir * np.dot(vdir, vup)

    vup *= 1 / np.linalg.norm(vup)
    right = np.cross(vup, vdir)

    rot_mat = np.transpose([right, vup, vdir])
    T = [
        list(rot_mat[0]) + [eye[0]],
        list(rot_mat[1]) + [eye[1]],
        list(rot_mat[2]) + [eye[2]],
        [0, 0, 0, 1],
    ]
    return np.array(T)


def gen_task_space():
    """Generates discrete set of poses to form the task space.

    Generates discrete set of poses, manually defined here as uniform grid facing into the world -z direction with 45 deg offsets.

    """
    poses = []
    ats = [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
    ]
    up_vector = [-1.0, 0.0, 0.0]

    step = 0.1
    pos_x_list = np.arange(0.25, 0.85 + step, step)
    pos_y_list = np.arange(-0.45, 0.45 + step, step)
    pos_z_list = np.arange(0.15, 0.45 + step, step)

    for pos_x in pos_x_list:
        for pos_y in pos_y_list:
            for pos_z in pos_z_list:
                for at_offset in ats:
                    eye = [pos_x, pos_y, pos_z]
                    # 0.001 is because IKFast solution is singular for poses pointing directly in z axis
                    at = [pos_x + 0.001, pos_y - 0.001, pos_z]
                    at = [x + y for x, y in zip(at, at_offset)]
                    T = transform_lookat(at, eye, up_vector)
                    poses.append(T)
    poses = np.array(poses)
    return poses


if __name__ == "__main__":
    import trimesh
    import matplotlib.pyplot as plt
    from pytransform3d.transformations import plot_transform
    from pytransform3d.plot_utils import make_3d_axis

    H1 = np.eye(4)
    H1[0:3, 3] = [1.0, 0.0, 0.0]
    H2 = np.eye(4)
    H2[0:3, 3] = [1.0, 0.0, 0.1]
    print("Split error:", se3_error_split(H1, H2))
    print("Total error:", se3_error(H1, H2, w_rot=10.0))
    print("Log error:", se3_error_log(H1, H2))

    # Hlist = poses_a()
    # Hlist = poses_b()
    # Hlist = poses_c()
    # Hlist = poses_d()
    Hlist = gen_task_space()
    print(f"==>> Hlist.shape: \n{Hlist.shape}")

    scene = trimesh.Scene()
    plane = trimesh.creation.box(extents=(4, 4, 0.01))
    plane.visual.face_colors = [200, 200, 200, 80]
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=2)
    box = trimesh.creation.box(extents=(4, 4, 4))
    box.visual.face_colors = [100, 150, 255, 40]
    scene.add_geometry(plane)
    scene.add_geometry(box)
    scene.add_geometry(axis)
    # points = Hlist[:, :3, 3]
    # point_cloud = trimesh.points.PointCloud(points, colors=[255, 0, 0, 255])
    # scene.add_geometry(point_cloud)
    for H in Hlist:
        axis = trimesh.creation.axis(origin_size=0.02, transform=H, axis_length=0.05, axis_radius=0.005)
        scene.add_geometry(axis)
    scene.show()

    # DIST = se3_error_pairwise_distance(Hlist, w_rot=1.0)
    # print(f"==>> DIST: \n{DIST}")

    # # raise
    # Hse3logerr = np.array([se3_log(H) for H in Hlist])  # shape (N,6)
    # # Normalize so translation/rotation are comparable:
    # Hse3logerr[:, :3] /= np.std(Hse3logerr[:, :3]) + 1e-8
    # Hse3logerr[:, 3:] /= np.std(Hse3logerr[:, 3:]) + 1e-8
    # print(f"==>> Hse3logerr: \n{Hse3logerr}")

    # mode = ["DBSCAN", "GMM_BIC"][0]
    # if mode == "DBSCAN":
    #     labels, cluster_id, num_clusters = dbscan_clustering(Hse3logerr)
    # if mode == "GMM_BIC":
    #     labels, cluster_id, num_clusters = gmm_bic_clustering(Hse3logerr)

    # print(f"==>> labels: \n{labels}")
    # print(f"==>> cluster_id: \n{cluster_id}")
    # print(f"==>> num_clusters: \n{num_clusters}")

    # Hmeans = {i: None for i in cluster_id if i != -1}
    # for i in Hmeans:
    #     Hmeans[i] = se3_mean(Hlist[labels == i])

    # nbrs = NearestNeighbors(n_neighbors=10).fit(Hse3logerr)
    # dists, indices = nbrs.kneighbors(Hse3logerr)
    # print(f"==>> dists: \n{dists}")
    # print(f"==>> indices: \n{indices}")
    # idx = 5
    # H0 = Hlist[idx]
    # H0neigh = Hlist[indices[idx]]

    # ----------------------------- Visualization -----------------------------
    # ax = make_3d_axis(1)
    # plot_transform(ax, name="world")
    # for i, H in Hmeans.items():
    #     plot_transform(ax, H, s=0.1, name=f"Hmean_{i}")
    # for i, H in enumerate(Hlist):
    #     plot_transform(ax, H, s=0.05, name=f"Cls {labels[i]}")
    # for hn in H0neigh:
    #     ax.plot(
    #         [H0[0, 3], hn[0, 3]],
    #         [H0[1, 3], hn[1, 3]],
    #         [H0[2, 3], hn[2, 3]],
    #         "k--",
    #         alpha=0.5,
    #     )
    # ax.set_title("SE(3) Error Metrics")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title(f"{mode} found {num_clusters} clusters")
    # ax.set_box_aspect([1, 1, 1])
    # plt.show()
