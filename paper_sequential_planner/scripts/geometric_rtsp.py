import os
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_config import (
    weighted_nan_euclidean_distances,
    weighted_nan_euclidean_squared_distances,
    weighted_nan_max_joint_diff_distances,
)
import pandas as pd

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


def Qfilter_ClusterTSP(Q, Qs, tmap, W=None):
    """
    ASMETRANonMECH2021, A novel clustering-based Spatially Const RTSP, Wong

    Filter candidate configurations by dissimilarity to home and all nodes.
    dissimilarity phi = bi * 𝛿_home + (1 - bi) * 𝛿_all
    Until there is one left for each task.
    The measuring metric in paper is the weighted euclidean squared distance.
    bias = bi + bihome = 1

    Q: node
    Qs: node validity from collision check (important)
    tmap: mapping dict
    W: weights for each joint
    """
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    ntasks = Q.shape[0]
    E = Eest_weighted_euclidean_squared(Q, Qs, W=W, tmap=tmap)

    # dissimilarity of all nodes to home 𝛿_home
    Qdelta_to_home = np.ones((ntasks, num_sols)) * np.inf
    for idx, (i, j) in enumerate(task_to_nn_pair):
        if i != 0:  # ignore the node that is not home
            continue
        # i == 0 is home
        E0 = E[idx, 0, :]  # distance from home to nodes in taskj (top row only)
        Qdelta_to_home[j, :] = E0

    # dissimilarity of each node to all other nodes 𝛿_all
    Qdelta_to_all = np.zeros((ntasks, num_sols))
    nQsvalid = np.sum(Qs)  # number of valid nodes per task
    for idx, (i, j) in enumerate(task_to_nn_pair):
        if i == 0:  # ignore the node that is home
            continue

        # sum over value that is not np.inf
        # do one for i
        Eij = E[idx]
        D = np.sum(Eij, where=np.isfinite(Eij), axis=1)  # sum in row for i->j
        Qdelta_to_all[i, :] += D

        # do one for j
        D = np.sum(Eij, where=np.isfinite(Eij), axis=0)  # sum in col for j->i
        Qdelta_to_all[j, :] += D

    # compute the mean dissimilarity to all other nodes
    Qdelta_to_all_mean = Qdelta_to_all / nQsvalid

    # dissimilarity phi = bi * 𝛿_home + (1 - bi) * 𝛿_all
    # the paper use 0.9 * weighted_avg + 0.1 * home_D
    phi = 0.9 * Qdelta_to_all_mean + 0.1 * Qdelta_to_home

    # method1: select the node with the minimum phi for each task only
    Qvalididx = np.argmin(phi, axis=1)  # the index 0 home is selected auto
    Qvalid = np.full((ntasks, num_sols, 1), False, dtype=bool)
    Qvalid[np.arange(ntasks), Qvalididx] = True

    # method2: iterative selection of nodes as in the paper
    # m = max(1, math.floor(math.log(len(Qi))))

    # print debug info
    print(f"==>> nQsvalid: \n{nQsvalid}")
    print(f"==>> Qdelta_to_home: \n{Qdelta_to_home}")
    print(f"==>> Qdelta_to_all: \n{Qdelta_to_all}")
    print(f"==>> Qdelta_to_all_mean: \n{Qdelta_to_all_mean}")
    print(f"==>> phi: \n{phi}")
    return Qvalid


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
       : implicitly provide infeasible check via np.inf in the edges
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
        : implicitly provide infeasible check via np.inf in the edges
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


def Eest_weighted_euclidean_squared(Q, Qs, W, tmap):
    """
    Input:
    Q: nodes
    Qs: nodes validity
    tmap: mapping dict
    W: weight for each joint in form of relative displacement in taskspace

    Compute:
    Consider every edges to be valid

    Output:
    Eweusq: edges heuristic distance based on weighted euclidean distance
        : implicitly provide infeasible check via np.inf in the edges
    """
    # get mapping
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]

    num_sols = Q.shape[1]
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols), dtype=dtype)
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = weighted_nan_euclidean_squared_distances(Q[i], Q[j], w=W)

    Estate = np.ones_like(E, dtype=bool)  # True/False mask
    # if Qs is not valid, then E is also invalid
    for idx, (i, j) in enumerate(task_to_nn_pair):
        QIJ = np.outer(Qs[i], Qs[j])  # (num_sols, num_sols)
        Estate[idx] = Estate[idx] & QIJ  # update E state

    Eweusq = np.where(Estate, E, np.inf)
    return Eweusq


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
        : implicitly provide infeasible check via np.inf in the edges
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


def Qtour_RoboTSP_layer_search(Ttour, E, tmap):
    """
    RoboTSP process of selecting the best q for each task given the pre-computed tour.

    Input:
    Ttour: the tour of tasks, with loop back to start
    E: edges cost matrix, must provide infeasible check via np.inf in the edges
    tmap: mapping dict
    """
    # get mapping
    task_to_nn_dict = tmap["task_to_nn_dict"]
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]
    ntasks = len(task_to_nn_dict.keys())  # already include start

    # verify tour
    # tour must start from 0 and end at 0 / loop back
    # the length of tour must be equal to the number of tasks + 2 (start and end)
    if Ttour[0] != 0 or Ttour[-1] != 0:
        raise ValueError("Tour must start and end at 0 (loop back).")
    if len(Ttour) != ntasks + 1:
        raise ValueError(
            f"Tour length must be equal to number of tasks + 2 (start and end). Expected {ntasks + 1}, got {len(Ttour)}."
        )

    # variables
    nQ = E.shape[1]  # number of candidate configurations per task
    nlayers = len(Ttour)  # number of tasks in the tour (including start and end)
    best_cost = np.full((nlayers, nQ), np.inf)
    best_cost[0, 0] = 0  # only the initial configuration is a valid source
    best_parent = np.full((nlayers, nQ), -1, dtype=int)

    # dynamic programming to find the best configuration for each task in the tour
    for i in range(nlayers - 1):
        transpose = False
        prev_i = Ttour[i]
        curr_i = Ttour[i + 1]

        if prev_i > curr_i:
            prev_i, curr_i = curr_i, prev_i  # swap to ensure prev_i < curr_i
            transpose = True  # the distance matrix is transposed

        try:  # make sure the pair exists in the mapping
            # get the index of dist mat
            I = task_to_nn_pair.index((prev_i, curr_i))
        except ValueError:
            raise ValueError(f"No dist mat found for pair ({prev_i}, {curr_i})")

        # distance matrix must provide infeasible check via np.inf in the edges
        Eij = E[I] if not transpose else E[I].T

        prev_cost = best_cost[i][:, np.newaxis]  # reshape (nQ, 1) for broadcasting
        candidate_cost = prev_cost + Eij  # addative cost to previous cost

        # find the best parent and cost for each configuration in the current task
        best_parent[i + 1] = np.argmin(candidate_cost, axis=0)
        best_cost[i + 1] = np.min(candidate_cost, axis=0)

    # Last layer cost
    if not np.isfinite(best_cost[-1]).any():
        raise RuntimeError("No feasible path found")

    def recover_path(goal_id):
        # backtrack to find the best tour
        # Qtour is task id dependent, its index is only from 0 to nQ, not flattened id
        Qtour = [goal_id]
        for layer in range(nlayers - 1, 0, -1):
            goal_id = best_parent[layer, goal_id]
            if goal_id < 0:
                raise RuntimeError(f"Broken parent chain at layer {layer}")
            Qtour.append(int(goal_id))
        Qtour = np.array(Qtour[::-1])  # reverse the path to get the correct order
        return Qtour

    goal_id = int(np.argmin(best_cost[-1]))
    Qtour = recover_path(goal_id)

    # debug print
    print(f"==>> Qtour: \n{Qtour} with cost: {best_cost[-1, goal_id]}")
    return Qtour


def Qtour_RoboTSP_layer_search_topK(Ttour, E, tmap, K):
    """
    RoboTSP process of selecting the best q for each task given the pre-computed tour.

    Input:
    Ttour: the tour of tasks, with loop back to start
    E: edges cost matrix, must provide infeasible check via np.inf in the edges
    tmap: mapping dict
    K: number of top configurations to keep
    """
    # get mapping
    task_to_nn_dict = tmap["task_to_nn_dict"]
    task_to_nn_pair = tmap["task_to_nn_pair"]
    task_to_nn_pair_len = tmap["task_to_nn_pair_len"]
    ntasks = len(task_to_nn_dict.keys())  # already include start

    # verify tour
    # tour must start from 0 and end at 0 / loop back
    # the length of tour must be equal to the number of tasks + 2 (start and end)
    if Ttour[0] != 0 or Ttour[-1] != 0:
        raise ValueError("Tour must start and end at 0 (loop back).")
    if len(Ttour) != ntasks + 1:
        raise ValueError(
            f"Tour length must be equal to number of tasks + 2 (start and end). Expected {ntasks + 1}, got {len(Ttour)}."
        )

    # variables
    nQ = E.shape[1]  # number of candidate configurations per task
    nlayers = len(Ttour)  # number of tasks in the tour (including start and end)
    best_cost = np.full((nlayers, nQ, K), np.inf)
    best_cost[0, 0, 0] = 0  # only the initial configuration is a valid source
    best_parent = np.full((nlayers, nQ, K), -1, dtype=int)
    best_parent_rank = np.full((nlayers, nQ, K), -1, dtype=int)  # track parentrank

    # dynamic programming to find the best configuration for each task in the tour
    for i in range(nlayers - 1):
        transpose = False
        prev_i = Ttour[i]
        curr_i = Ttour[i + 1]

        if prev_i > curr_i:
            prev_i, curr_i = curr_i, prev_i  # swap to ensure prev_i < curr_i
            transpose = True  # the distance matrix is transposed

        try:  # make sure the pair exists in the mapping
            # get the index of dist mat
            I = task_to_nn_pair.index((prev_i, curr_i))
        except ValueError:
            raise ValueError(f"No dist mat found for pair ({prev_i}, {curr_i})")

        # distance matrix must provide infeasible check via np.inf in the edges
        Eij = E[I] if not transpose else E[I].T

        prev_cost = best_cost[i][:, :, None]
        candidate_cost = prev_cost + Eij[:, None, :]  # (nQ, K, 1)  # (nQ, 1, nQ)

        # reshape:
        # (previous_node, previous_rank, current_node)
        # -> (previous_node * previous_rank, current_node)
        flat_cost = candidate_cost.reshape(nQ * K, nQ)

        # For each current node, get K smallest candidates
        # partition is much faster than fully sorting everything.
        kk = min(K, flat_cost.shape[0])
        idx = np.argpartition(flat_cost, kth=kk - 1, axis=0)[:kk]

        selected_cost = np.take_along_axis(flat_cost, idx, axis=0)
        order = np.argsort(selected_cost, axis=0)  # Sort those K candidates
        selected_cost = np.take_along_axis(selected_cost, order, axis=0)
        idx = np.take_along_axis(idx, order, axis=0)

        # Decode flattened parent index
        parent_node = idx // K
        parent_rank = idx % K

        # Store
        best_cost[i + 1, :, :kk] = selected_cost.T
        best_parent[i + 1, :, :kk] = parent_node.T
        best_parent_rank[i + 1, :, :kk] = parent_rank.T

    # Last layer cost
    final_cost = best_cost[-1].reshape(-1)
    valid = np.isfinite(final_cost)
    if not valid.any():
        raise RuntimeError("No feasible path found")

    valid_indices = np.where(valid)[0]
    order = valid_indices[np.argsort(final_cost[valid_indices])]
    order = order[:K]

    def recover_path(final_flat_index):
        goal_node = final_flat_index // K
        goal_rank = final_flat_index % K
        Qtour = np.empty(nlayers, dtype=int)
        node = goal_node
        rank = goal_rank
        Qtour[-1] = node
        for layer in range(nlayers - 1, 0, -1):
            parent_node = best_parent[layer, node, rank]
            parent_rank = best_parent_rank[layer, node, rank]
            if parent_node < 0 or parent_rank < 0:
                raise RuntimeError(f"Broken parent chain at layer {layer}")
            node = parent_node
            rank = parent_rank
            Qtour[layer - 1] = node
        return Qtour

    Qtourlist = []
    for idx in order:
        Qtour = recover_path(idx)
        Qtourlist.append(Qtour)
        # cost = final_cost[idx]
    Qtourlist = np.array(Qtourlist)

    # debug print
    print(f"==>> Qtourlist:")
    for ord, cost in zip(order, final_cost[order]):
        print(f"{Qtourlist[ord]} with cost: {cost}")
    return Qtourlist
