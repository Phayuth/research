import os
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_config import (
    weighted_nan_euclidean_distances,
    weighted_nan_max_joint_diff_distances,
)

dtype = np.float32


def Qfilter_R(Q, q, Qs, r):
    """
    My own thoughts

    Filter candidate configurations that are too far from initial configuration
    using a simple radius threshold in cspace distance.

    !Important: make sure that the selected configurations are also valid from Qs.
    !If the selected configurations have its edge in minimum cost but the node is invalid,
    !then we selected the next minimum cost edge that is valid.

    Q: node
    q: initial configuration
    Qs: node validity from collision check (important)
    r: radius threshold
    """
    ntasks_rech, n_ik, dof = Q.shape
    Q_flat = Q.reshape(ntasks_rech * n_ik, dof)
    dist = nan_euclidean_distances(Q_flat, q.reshape(1, -1))
    q_valid = dist.flatten() <= r
    Qvalid = q_valid.reshape(ntasks_rech, n_ik)
    Qvalid = Qvalid[:, :, None]  # just add a dummy dimension
    Qvalid = Qvalid & Qs  # ensure nodes is valid from collision check

    # print(f"---------------------------------------------------------")
    nQredpt = np.sum(Qvalid, axis=1)
    n_selected = np.sum(nQredpt)
    n_total = np.prod(Qvalid.shape)
    # print(f"==>> Qfilter R debug Info")
    # print(f"==>> selected {n_selected} / {n_total} configurations")
    # print(f"==>> selected_rate: {n_selected / n_total}")
    # print(f"---------------------------------------------------------")
    return Qvalid


def Qfilter_similarity(Q, q, Qs, thresh, W):
    """
    CASE2022, An Efficient Approach for solving RTSP, Li

    Filter candidate configurations that are too far from initial configuration
    Weighted euclidean distance to initial config. Now i dont have the weight yet.
    Bigger val mean closer to qinit mean very little Q selected.
    Samaller val mean farther to qinit mean more Q selected.

    Q: node
    q: initial configuration
    Qs: node validity from collision check (important)
    thresh: threshold for selection
    W: weights for each joint
    """
    ntasks_rech, n_ik, dof = Q.shape
    Q_flat = Q.reshape(ntasks_rech * n_ik, dof)
    dist = weighted_nan_euclidean_distances(Q_flat, q.reshape(1, -1), w=W)
    del_sim = 1.0 / (dist + 0.001)  # avoid division by zero
    phi_opt = del_sim / np.nansum(del_sim)  # normalize to sum to 1
    q_valid = phi_opt >= thresh
    Qvalid = q_valid.reshape(ntasks_rech, n_ik)
    Qvalid = Qvalid[:, :, None]  # just add a dummy dimension
    Qvalid = Qvalid & Qs  # ensure nodes is valid from collision check

    # threshold = thresh_mult * (optimal_val_max - optimal_val_min) + optimal_val_min
    # print(f"---------------------------------------------------------")
    nQredpt = np.sum(Qvalid, axis=1)
    n_selected = np.sum(nQredpt)
    n_total = np.prod(Qvalid.shape)
    phi_opt_min = np.nanmin(phi_opt)
    phi_opt_max = np.nanmax(phi_opt)
    # print(f"==>> Qfilter similarity debug Info")
    # print(f"==>> optimal values: min={phi_opt_min}, max={phi_opt_max}")
    # print(f"==>> selected {n_selected} / {n_total} configurations")
    # print(f"==>> selected_rate: {n_selected / n_total}")
    # print(f"---------------------------------------------------------")
    return Qvalid


def Qfilter_nn2c(Q, Qs, tmap):
    """
    GECCO2017 & Operational Research 2019, A Pre-processing reduction GTSP, Mehdi

    Filter candidate configurations such that every cluster pair has at least
    one valid edge between them.

    !Important: make sure that the selected configurations are also valid from Qs.
    !If the selected configurations have its edge in minimum cost but the node is invalid,
    !then we selected the next minimum cost edge that is valid.

    Q: node
    Qs: node validity from collision check (important)
    tmap: mapping dict
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols))
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = nan_euclidean_distances(Q[i], Q[j])

    ntasks_rech, n_ik, dof = Q.shape
    Qvalid = np.zeros((ntasks_rech, n_ik), dtype=bool)
    # Qminval = np.zeros_like(Q)
    for idx, (i, j) in enumerate(task_to_nn_pair):
        Esij = np.outer(Qs[i], Qs[j])  # ensure nodes is valid from collision check
        Eij = np.where(Esij, E[idx], np.nan)  # filter out invalid edges value
        UV = np.unravel_index(np.nanargmin(Eij), Eij.shape)
        qu = np.zeros(n_ik, dtype=bool)
        qu[UV[0]] = True
        qv = np.zeros(n_ik, dtype=bool)
        qv[UV[1]] = True
        Qvalid[i] = Qvalid[i] | qu
        Qvalid[j] = Qvalid[j] | qv

    return Qvalid[:, :, None]  # add a dummy dimension


def Qfilter_Knn2c(Q, Qs, k, tmap):
    """
    The same as Qfilter_nn2c, but we select the K nearest neighbors

    Q: node
    Qs: node validity from collision check (important)
    k: number of nearest neighbors to select
    tmap: mapping dict
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols))
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = nan_euclidean_distances(Q[i], Q[j])

    ntasks_rech, n_ik, dof = Q.shape
    Qvalid = np.zeros((ntasks_rech, n_ik), dtype=bool)
    for idx, (i, j) in enumerate(task_to_nn_pair):
        Esij = np.outer(Qs[i], Qs[j])  # ensure nodes is valid from collision check
        Eij = np.where(Esij, E[idx], np.nan)  # filter out invalid edges value
        # Get the k nearest neighbors for each node
        knn_indices = np.argpartition(Eij, k, axis=None)[:k]
        knn_mask = np.zeros_like(Eij, dtype=bool)
        knn_mask.flat[knn_indices] = True
        Qvalid[i] = Qvalid[i] | knn_mask.any(axis=1)
        Qvalid[j] = Qvalid[j] | knn_mask.any(axis=0)

    return Qvalid[:, :, None]  # add a dummy dimension


def Qfilter_Dnn2c(Q, Qs, d, tmap):
    """
    The same as Qfilter_nn2c, but we select the in Distance nearest neighbors

    Q: node
    Qs: node validity from collision check (important)
    d: distance threshold for nearest neighbors
    tmap: mapping dict
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols))
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = nan_euclidean_distances(Q[i], Q[j])

    ntasks_rech, n_ik, dof = Q.shape
    Qvalid = np.zeros((ntasks_rech, n_ik), dtype=bool)
    for idx, (i, j) in enumerate(task_to_nn_pair):
        Esij = np.outer(Qs[i], Qs[j])  # ensure nodes is valid from collision check
        Eij = np.where(Esij, E[idx], np.nan)  # filter out invalid edges value
        # Get the indices of elements less than or equal to d
        dnn_mask = Eij <= d
        Qvalid[i] = Qvalid[i] | dnn_mask.any(axis=1)
        Qvalid[j] = Qvalid[j] | dnn_mask.any(axis=0)

    return Qvalid[:, :, None]  # add a dummy dimension


def Qfilter_Favor(Q, Qs, tmap):
    """
    Favor the first solution of each task, and select the nearest neighbor for each task pair.

    Q: node
    Qs: node validity from collision check (important)
    tmap: mapping dict
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]
    pass


def Eest_colfree(Q, Qs, cmax_d, tmap):
    """
    Input:
    Q: nodes
    Qs: nodes validity
    tmap: mapping dict

    Compute:
    If provide cmax_d, then only estimate edges that are cost less than 2pi, otherwise invalid (np.inf)
    If no path between two nodes, then the edge is also invalid (np.inf)

    Output:
    Ecf: edges collision-free distance
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols), dtype=dtype)
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = nan_euclidean_distances(Q[i], Q[j])

    if cmax_d is not None:
        Estate = E <= cmax_d  # True/False mask dist over cmax_d
        # if Qs is not valid, then E is also invalid
        for idx, (i, j) in enumerate(task_to_nn_pair):
            QIJ = np.outer(Qs[i], Qs[j])  # (num_sols, num_sols)
            Estate[idx] = Estate[idx] & QIJ  # update E state
    else:
        Estate = np.ones_like(E, dtype=bool)  # True/False mask

    # from the Estate, we estimation the collision-free distance
    # If Estate is True then we have distance valid
    # If Estate is False then we have distance as np.inf
    # For now we use fake cost
    Ecf = np.where(Estate, E, np.inf) + np.random.random(E.shape) * 1e-6
    return Ecf


def Eest_weighted_euclidean(Q, Qs, W, tmap):
    """
    Input:
    Q: nodes
    Qs: nodes validity
    tmap: mapping dict
    W: weight for each joint in form of relative displacement in taskspace

    Compute:
    Consider every edges to be valid

    Output:
    Eweu: edges heuristic distance based on weighted euclidean distance
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols), dtype=dtype)
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = weighted_nan_euclidean_distances(Q[i], Q[j], w=W)

    Estate = np.ones_like(E, dtype=bool)  # True/False mask
    # if Qs is not valid, then E is also invalid
    for idx, (i, j) in enumerate(task_to_nn_pair):
        QIJ = np.outer(Qs[i], Qs[j])  # (num_sols, num_sols)
        Estate[idx] = Estate[idx] & QIJ  # update E state

    Eweu = np.where(Estate, E, np.inf)
    return Eweu


def Eest_weighted_max_joint_diff(Q, Qs, W, tmap):
    """
    Input:
    Q: nodes
    Qs: nodes validity
    tmap: mapping dict
    W: weight for each joint in form of joint velocity

    Compute:
    Consider every edges to be valid

    Output:
    Ewmj: edges heuristic distance based on max joint difference
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols), dtype=dtype)
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = weighted_nan_max_joint_diff_distances(Q[i], Q[j], w=W)

    Estate = np.ones_like(E, dtype=bool)  # True/False mask
    # if Qs is not valid, then E is also invalid
    for idx, (i, j) in enumerate(task_to_nn_pair):
        QIJ = np.outer(Qs[i], Qs[j])  # (num_sols, num_sols)
        Estate[idx] = Estate[idx] & QIJ  # update E state

    Ewmj = np.where(Estate, E, np.inf)
    return Ewmj


def search_dijkstra_graph(E, tmap):
    import networkx as nx

    NXG = nx.Graph()

    NXG.add_weighted_edges_from()
    pass
