import os
import numpy as np
from scipy.spatial.transform import Rotation as R

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

"""
Distance metrics
"""


def position_pairwise_distances(Hlist):
    """
    H: (N, 4, 4)
    return: (N, N) pairwise position distance
    """
    t = Hlist[:, :3, 3]  # (N, 3)
    dt = t[:, None, :] - t[None, :, :]  # (N, N, 3)
    D = np.sqrt(np.sum(dt**2, axis=-1))  # (N, N)
    return D


def SE3_pairwise_distances(Hlist, w_rot=1.0):
    """
    H: (N, 4, 4)
    return: (N, N) pairwise SE(3) distance
    Use w_rot equal to 0.0 if you want to ignore rotation.
    """
    R = Hlist[:, :3, :3]  # (N, 3, 3)
    t = Hlist[:, :3, 3]  # (N, 3)

    dt = t[:, None, :] - t[None, :, :]  # (N, N, 3)
    et2 = np.sum(dt**2, axis=-1)  # (N, N)

    # (N, N, 3, 3)
    R_rel = np.matmul(R.transpose(0, 2, 1)[:, None, :, :], R[None, :, :, :])

    tr = np.trace(R_rel, axis1=-2, axis2=-1)  # (N, N)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (N, N)

    D = np.sqrt(et2 + (w_rot * theta) ** 2)
    return D


"""
Taskspace Correlation based on all neighbors
All tasks are connected to all other tasks.
This is the naive version, which is not efficient and not scalable.
"""


def Naive_task_space_correlation(H):
    """
    H: (N, 4, 4)
    return: taskspace correlation mapping
    """
    N = H.shape[0]
    task_to_nn_dict = {i: [j for j in range(N) if j != i] for i in range(N)}
    task_to_nn_pair = set()
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            task_to_nn_pair.add((a, b))
    task_to_nn_pair = sorted(task_to_nn_pair)
    task_to_nn_pair_len = len(task_to_nn_pair)

    tspace_mapping = {
        "task_to_nn_dict": task_to_nn_dict,
        "task_to_nn_pair": task_to_nn_pair,
        "task_to_nn_pair_len": task_to_nn_pair_len,
    }
    return tspace_mapping


"""
Taskspace Correlation based on k-NN and r-NN with pure SE(3) distance metric
Mutual Discussion
k-NN: directed graph (not mutual)
r-NN: undirected graph (mutual, under symmetric metric)
So we most likely dont find mutual k-NN
"""


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
    Dse3 = SE3_pairwise_distances(H, w_rot=w_rot)
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
    Dse3 = SE3_pairwise_distances(H, w_rot=wse3_rot)
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
