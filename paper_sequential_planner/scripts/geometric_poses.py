import os
import numpy as np
from scipy.spatial.transform import Rotation as R

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

"""
Taskspace Correlation based on k-NN and r-NN with pure SE(3) distance metric
Mutual Discussion
k-NN: directed graph (not mutual)
r-NN: undirected graph (mutual, under symmetric metric)
So we most likely dont find mutual k-NN
"""


def se3_pairwise_distances(H, w_rot=1.0):
    """
    H: (N, 4, 4)
    return: (N, N) pairwise SE(3) distance
    Use w_rot equal to 0.0 if you want to ignore rotation.
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


def knn_from_distance(D, k):
    # ignore self-distance by setting diagonal large
    D = D.copy()
    np.fill_diagonal(D, np.inf)

    idx = np.argpartition(D, k, axis=1)[:, :k]  # (N, k)

    # optional: sort neighbors by distance
    row_idx = np.arange(D.shape[0])[:, None]
    sorted_order = np.argsort(D[row_idx, idx], axis=1)
    idx = idx[row_idx, sorted_order]

    return idx.tolist()  # indices of k nearest per row


def rnn_from_distance(D, radius):
    neighbors = []
    for i in range(D.shape[0]):
        idx = np.where(D[i] < radius)[0]
        idx = idx[idx != i]  # remove self
        neighbors.append(idx.tolist())
    return neighbors


def KRNN_task_space_correlation(H, w_rot, nnr, nnk):
    Dse3 = se3_pairwise_distances(H, w_rot=w_rot)
    nn_k = knn_from_distance(Dse3, k=nnk)
    nn_r = rnn_from_distance(Dse3, radius=nnr)
    nn_union = []
    for i in range(Dse3.shape[0]):
        union_set = set(nn_r[i]) | set(nn_k[i])
        nn_union.append(sorted(union_set))
    nn_dist = []
    for i in range(len(nn_union)):
        dists = [Dse3[i, j].item() for j in nn_union[i]]
        nn_dist.append(dists)

    nn_count = [len(n) for n in nn_union]

    tspace_coorrelation = {
        "nn_union": nn_union,
        "nn_dist": nn_dist,
        "nn_count": nn_count,
        "nn_r": nn_r,
        "nn_k": nn_k,
    }

    # *mapping construction
    nn_union = tspace_coorrelation["nn_union"]

    task_to_nn_dict = {}
    for i in range(len(nn_union)):
        for j in nn_union[i]:
            task_to_nn_dict[i] = task_to_nn_dict.get(i, []) + [j]

    # unique undirected edges in canonical order: (i, j) with i < j
    task_to_nn_pair = set()
    for i in range(len(nn_union)):
        for j in nn_union[i]:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            task_to_nn_pair.add((a, b))

    task_to_nn_pair = sorted(task_to_nn_pair)
    task_to_nn_pair_len = len(task_to_nn_pair)  # number of unique undirected pairs

    tspace_mapping = {
        "task_to_nn_dict": task_to_nn_dict,
        "task_to_nn_pair": task_to_nn_pair,
        "task_to_nn_pair_len": task_to_nn_pair_len,
    }
    return tspace_mapping


"""
Taskspace Correlation based on advanced Robotics Arm Metric
For Robotics Arm, Consider purely SE(3) distance metric is not enough.
We must consider extra factors like
- joint limits
- singularity
- manipulability
- collision
- redundancy
"""


def Advanced_task_space_correlation(H, Q, Qs, W):
    """
    H: (N, 4, 4) task space poses
    Q: (N, numik, dof) robot configurations
    Qs: (N, numik, 1) validity flags for each configuration
    W: weight factors
    """
    wse3_rot = W["wse3_rot"]
    Dse3 = se3_pairwise_distances(H, w_rot=wse3_rot)
    print(f"==>> Dse3.shape: \n{Dse3.shape}")


"""
# Pose format conversion
# x = (x,y,z) # shape (3,)
# X = (x,y,z, qx, qy, qz, qw) # shape (7,)
# Xlist = [(x,y,z, qx, qy, qz, qw), ...] # shape (N,7)
# H = [R|t] in SE(3) # shape (4,4)
# Hlist = [H1, H2, ...] # shape (N,4,4)
"""


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


def xlist_to_Xlist(xlist):
    """
    Repair pose format
    due to some robot dofs is not 6dof, we concat the pose with some dummy values
    X = (x,y,z, qx, qy, qz, qw) # shape (7,)
    H = [R|t] in SE(3) # shape (4,4)
    """
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


# *generate synthetic poses ----------------------------------------------
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


def poses_epGH():
    """Generates discrete set of poses to form the task space.

    Generates discrete set of poses, manually defined here as uniform grid facing into the world -z direction with 45 deg offsets.

    """

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
