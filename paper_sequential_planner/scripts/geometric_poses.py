import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


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


# *taskspace correlation ------------------------------------------------------
# k-NN: directed graph (not mutual)
# r-NN: undirected graph (mutual, under symmetric metric)
# so we most likely dont find mutual k-NN
def rnn_from_distance(D, radius):
    neighbors = []
    for i in range(D.shape[0]):
        idx = np.where(D[i] < radius)[0]
        idx = idx[idx != i]  # remove self
        neighbors.append(idx.tolist())
    return neighbors


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


def task_space_correlation(tspace_dist, nnr=0.15, nnk=5):
    nn_r = rnn_from_distance(tspace_dist, radius=nnr)
    nn_k = knn_from_distance(tspace_dist, k=nnk)
    nn_union = []
    for i in range(tspace_dist.shape[0]):
        union_set = set(nn_r[i]) | set(nn_k[i])
        nn_union.append(sorted(union_set))
    nn_dist = []
    for i in range(len(nn_union)):
        dists = [tspace_dist[i, j].item() for j in nn_union[i]]
        nn_dist.append(dists)

    nn_count = [len(n) for n in nn_union]

    tspace_coorrelation = {
        "nn_union": nn_union,
        "nn_dist": nn_dist,
        "nn_count": nn_count,
        "nn_r": nn_r,
        "nn_k": nn_k,
    }
    return tspace_coorrelation


def task_space_correlation_map(tspace_coorrelation):
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


def query_data_from_tspace_map(i, j, cspace_eudist, task_to_nn_pair):
    """
    I is task [from] id
    J is task [to] id
    query (I, J) and (J, I) give the same value
    but the shape is transposed to swap T1 and T2
    """
    if j < i:
        a, b = (i, j) if i < j else (j, i)
        idx = task_to_nn_pair.index((a, b))
        return cspace_eudist[idx].T  # transpose to swap T1 and T2
    else:
        a, b = (i, j) if i < j else (j, i)
        idx = task_to_nn_pair.index((a, b))
        return cspace_eudist[idx]


# *rack system ----------------------------------------------------------------


# *cluster algorithm ----------------------------------------------------------
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


# # Normalize so translation/rotation are comparable:
# Hse3logerr = np.array([se3_log(H) for H in Hlist])  # shape (N,6)
# Hse3logerr[:, :3] /= np.std(Hse3logerr[:, :3]) + 1e-8
# Hse3logerr[:, 3:] /= np.std(Hse3logerr[:, 3:]) + 1e-8
# mode = ["DBSCAN", "GMM_BIC"][0]
# if mode == "DBSCAN":
#     labels, cluster_id, num_clusters = dbscan_clustering(Hse3logerr)
# if mode == "GMM_BIC":
#     labels, cluster_id, num_clusters = gmm_bic_clustering(Hse3logerr)
# Hmeans = {i: None for i in cluster_id if i != -1}
# for i in Hmeans:
#     Hmeans[i] = se3_mean(Hlist[labels == i])


def se3_mean(Hs, max_iter=20):
    """
    Determine the mean pose of SE(3) from cluster of Hs
    """
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


# *pose format conversion ------------------------------------------------------
# x = (x,y,z) # shape (3,)
# X = (x,y,z, qx, qy, qz, qw) # shape (7,)
# Xlist = [(x,y,z, qx, qy, qz, qw), ...] # shape (N,7)
# H = [R|t] in SE(3) # shape (4,4)
# Hlist = [H1, H2, ...] # shape (N,4,4)


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


# *cspace --------------------------------------------------------------------
def brute_cspace_distance(Q):
    """
    Very easy but memory-heavy way to compute pairwise cspace distance by IK pairing
    Q is (ntasks_rech, n_ik*altcnf, dof)

    Return shape (ntasks_rech, ntasks_rech, n_ik*altcnf, n_ik*altcnf)
    accessing same id will give us dist to itself, which we dont need.
    always access different task id to get task-to-task distance, which we need
    """
    ntasks_rech, n_ik, dof = Q.shape
    _Qflat = Q.reshape(ntasks_rech * n_ik, dof)
    _cspace_eudist_flat = nan_euclidean_distances(_Qflat, _Qflat)
    cspace_eudist = _cspace_eudist_flat.reshape(
        ntasks_rech, n_ik, ntasks_rech, n_ik
    )
    cspace_eudist = cspace_eudist.transpose(0, 2, 1, 3)
    return cspace_eudist


def brute_cspace_min_distance(cspace_eudist):
    """ """
    _cspace_dist_inf = np.where(np.isnan(cspace_eudist), np.inf, cspace_eudist)
    cspace_task_min = _cspace_dist_inf.min(axis=(2, 3))
    cspace_task_min[~np.isfinite(cspace_task_min)] = np.nan

    min_flat_idx = np.argmin(
        _cspace_dist_inf.reshape(
            _cspace_dist_inf.shape[0], _cspace_dist_inf.shape[1], -1
        ),
        axis=2,
    )
    min_idx_2d = np.unravel_index(min_flat_idx, _cspace_dist_inf.shape[2:])
    cspace_task_min_idx = np.stack(min_idx_2d, axis=-1)

    cspace_task_min_values = {
        "cspace_task_min": cspace_task_min,
        "cspace_task_min_idx": cspace_task_min_idx,
    }
    return cspace_task_min_values


def task_to_task_configuration_interp(Qaik_rall, nintp=10):
    """
    Vectorized Interpolate between all pairs of task-reachable configurations,
    and set to nan if either of the pair is invalid.
    final shape (ntasks_rech, ntasks_rech, n_ik, n_ik, nintp, dof)

    [This memory-heavy interpolation, only be used for small number of tasks]
    Ex:
    tt_cspace_interp = task_to_task_configuration_interp(Qaik_rall, nintp=10)
    t0t1q0q0 = tt_cspace_interp[0, 1, 0, 0]
    give us interp bet/ first IK of task0 to first IK of task1.
    task id should not be the same
    """
    ntasks_rech, n_ik, dof = Qaik_rall.shape
    Q1 = Qaik_rall[:, None, :, None, None, :]  # (ntasks_rech,1,n_ik,1,1,dof)
    Q2 = Qaik_rall[None, :, None, :, None, :]  # (1,ntasks_rech,1,n_ik,1,dof)
    tau = np.linspace(0.0, 1.0, nintp, dtype=Qaik_rall.dtype)
    tau = tau[None, None, None, None, :, None]  # (1,1,1,1,nintp,1)
    # (ntasks_rech,ntasks_rech,n_ik,n_ik,nintp,dof)
    interp = (1.0 - tau) * Q1 + tau * Q2
    invalid_cfg = np.isnan(Qaik_rall).all(axis=-1)  # (ntasks_rech,n_ik)
    # (ntasks_rech,ntasks_rech,n_ik,n_ik)
    invalid_pair = invalid_cfg[:, None, :, None] | invalid_cfg[None, :, None, :]
    interp = np.where(invalid_pair[..., None, None], np.nan, interp)
    return interp


def filter_cspace_candidate_similar_to_qinit(Qaik_r, qinit, thresh_mult=0.08):
    """
    From paper CASE2022
    An Efficient Approach for solving RTSP Considering Spatial Constraint

    Filter candidate configurations that are too far from initial configuration
    Weighted euclidean distance to initial config. now i dont have the weight.
    thresh_mult is a hyperparam to control how aggressive the filtering is,
    between 0 and 1 of min to max ratio value.
    """
    ntasks_rech, n_ik, dof = Qaik_r.shape
    Qaik_r_flat = Qaik_r.reshape(ntasks_rech * n_ik, dof)
    dist = nan_euclidean_distances(Qaik_r_flat, qinit.reshape(1, -1))
    sim_val = 1.0 / (dist + 0.001)  # add small value to avoid div by zero
    optimal_val = sim_val / np.nansum(sim_val)  # use nansum to ignore nan
    optimal_val_min = np.nanmin(optimal_val)
    optimal_val_max = np.nanmax(optimal_val)
    threshold = thresh_mult * (optimal_val_max - optimal_val_min) + optimal_val_min
    get_Qind_inthreshold = optimal_val >= threshold
    Qin_sim = Qaik_r_flat[get_Qind_inthreshold.flatten()]
    selected_q = get_Qind_inthreshold.reshape(ntasks_rech, n_ik)
    selected_rate = np.sum(get_Qind_inthreshold) / get_Qind_inthreshold.size
    task_ids = np.where(get_Qind_inthreshold)[0] // n_ik
    qi_in_task = np.where(get_Qind_inthreshold)[0] % n_ik

    print(f"min: {optimal_val_min}, max: {optimal_val_max}, thresh: {threshold}")
    print(f"==>> selected {len(Qin_sim)} / {len(Qaik_r_flat)} configurations")
    print(f"==>> selected_rate: {selected_rate}")
    fd_sim = {
        "selected_q": selected_q[:, :, None],
        "Qin_sim": Qin_sim,
        "task_ids": task_ids + 1,  # task 0 is qstart, add 1 to start with first H
        "qi_in_task": qi_in_task,
    }
    return fd_sim


def filter_cspace_candidate_radius_to_qinit(Qaik_r, qinit, radius=2 * np.pi):
    """
    Filter candidate configurations that are too far from initial configuration
    using a simple radius threshold in cspace.
    """
    ntasks_rech, n_ik, dof = Qaik_r.shape
    Qaik_r_flat = Qaik_r.reshape(ntasks_rech * n_ik, dof)
    dist = nan_euclidean_distances(Qaik_r_flat, qinit.reshape(1, -1))
    q_valid = dist.flatten() <= radius
    q_valid_shape = q_valid.reshape(ntasks_rech, n_ik)
    q_valid_shape = q_valid_shape[:, :, None]  # just add a dummy dimension

    nQredpt = np.sum(q_valid_shape, axis=1)
    n_selected = np.sum(nQredpt)
    n_total = np.prod(q_valid_shape.shape)
    print(f"==>> selected {n_selected} / {n_total} configurations")
    print(f"==>> selected_rate: {n_selected / n_total}")
    return q_valid_shape


def filter_cspace_candidate_nn2c(Qaik_r, qinit):
    pass


def filter_cspace_edges():
    pass


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


if __name__ == "__main__":
    import trimesh

    # Hlist = poses_a()
    Hlist = poses_b()
    # Hlist = poses_c()
    # Hlist = poses_d()
    # Hlist = poses_epGH()
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
        axis = trimesh.creation.axis(
            origin_size=0.002, transform=H, axis_length=0.05, axis_radius=0.0008
        )
        scene.add_geometry(axis)
    scene.show()
