import os
import numpy as np
import time
import tqdm
import torch
from sklearn.metrics.pairwise import nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_torus import find_alt_config2
from paper_sequential_planner.scripts.geometric_poses import (
    H_to_X,
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    Naive_task_space_correlation,
    KRNN_task_space_correlation,
    Advanced_task_space_correlation,
)
from paper_sequential_planner.experiments.env_ur5e_sphere import (
    RobotUR5eKin,
    SceneUR5eSpherized,
    pick_task_poses,
)
from paper_sequential_planner.experiments.utilio import (
    check_number_Q,
    check_number_E,
    write_gtsp_file,
    read_gtsp_file,
    call_gtsp_glns_solver,
    write_tour_path,
)

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robkin = RobotUR5eKin()
scene = SceneUR5eSpherized()
planner = None  # placeholder for planner instance, not implemented yet

alt_num = 32
unique_sols = 8
num_sols = unique_sols * alt_num
dof = 6


def wspace_ik_extended(robot, Xtspace):
    # this is general from hardware, this has 32 redundant solutions
    limit6 = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-np.pi, np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )
    # if the table is under, q2 is limited to [-pi, 0]
    # limit6[1, 0] = -np.pi
    # limit6[1, 1] = 0

    ntasks = Xtspace.shape[0]
    Qaik = np.full((ntasks, num_sols, dof), np.nan)

    Htasks = Xlist_to_Hlist(Xtspace)
    for taski in range(ntasks):
        nik, q_sols = robot.solve_aik(Htasks[taski])
        if nik == 0:
            Qaik[taski] = np.nan
        for qi, q in enumerate(q_sols):
            q = q + 1e-2  # to avoid numerical issues in find_alt_config2
            alt_qs = find_alt_config2(q, limit6, filterOriginalq=False)
            Qaik[taski, qi * alt_num : (qi + 1) * alt_num] = alt_qs
    return Qaik


def queue_Qaik_batch_collision(Qaik, robscene):
    # Qaik shape: (ntasks, num_sols, dof)
    ntasks = Qaik.shape[0]
    Qmin_rep = np.empty((ntasks, unique_sols, dof))
    for taski in range(ntasks):
        for solj in range(unique_sols):
            i = solj * alt_num
            j = (solj + 1) * alt_num
            Q = Qaik[taski, i:j]  # (alt_num, dof)
            q = Q[0]
            Qmin_rep[taski, solj] = q
    Qmin_flat = Qmin_rep.reshape(-1, dof)  # (ntasks*unique_sols, dof)
    col_states = robscene.collision_check(Qmin_flat).detach().cpu().numpy()
    col_states_rep = col_states.reshape(ntasks, unique_sols)
    Qaik_rep = np.repeat(col_states_rep[:, :, np.newaxis], alt_num, axis=2)
    Qaik_rep_col = Qaik_rep.reshape(ntasks, num_sols)  # (ntasks, num_sols)
    return Qaik_rep_col


def wspace_ik_validity_extended(Qaik, robscene):
    limit6 = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-np.pi, np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )
    Qaik_rep_col = queue_Qaik_batch_collision(Qaik, robscene)  # batch colcheck
    ntasks = Qaik.shape[0]
    num_sols = Qaik.shape[1]
    eps = 1e-9
    Qaik_valid = np.full((ntasks, num_sols, 1), np.nan)
    for taski in range(ntasks):
        for solj in range(num_sols):
            q = Qaik[taski, solj]
            if np.isnan(q).any():
                Qaik_valid[taski, solj] = -1  # No solution
            else:
                isCollsion = Qaik_rep_col[taski, solj]
                if isCollsion:
                    Qaik_valid[taski, solj] = -2  # In collision
                else:
                    is_in_limit = np.all(
                        (q >= limit6[:, 0] - eps) & (q <= limit6[:, 1] + eps)
                    )
                    if not is_in_limit:
                        Qaik_valid[taski, solj] = -3  # Out of limits
                    else:
                        Qaik_valid[taski, solj] = 1  # Valid solution
    return Qaik_valid.astype(int)


# preliminary input data processing
qinit = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])
Xinit = H_to_X(robkin.solve_fk(qinit))
H = pick_task_poses()
X = Hlist_to_Xlist(H)
ntasks = X.shape[0]
Qik = wspace_ik_extended(robkin, X)
Qikstate = wspace_ik_validity_extended(Qik, scene)

# filter out the unreachable tasks
Xunreach = np.all(Qikstate != 1, axis=1).flatten()
X_reach = X[~Xunreach]
Qik_reach = Qik[~Xunreach]
Qikstate_reach = Qikstate[~Xunreach]

# concat the init to the reachable tasks
qinit_ = np.full((1, Qik_reach.shape[1], Qik_reach.shape[2]), np.nan)
qinit_[0, 0] = qinit
Qik_reach_init = np.vstack((qinit_, Qik_reach))  # init & ntasks
qinit_state_ = np.full((1, Qikstate_reach.shape[1], Qikstate_reach.shape[2]), -1)
qinit_state_[0, 0] = 1
Qikstate_reach_init = np.vstack((qinit_state_, Qikstate_reach))  # init & ntasks
Qikstate_reach_init = np.where(Qikstate_reach_init == 1, True, False)  # T/F mask
X_reach_init = np.vstack((Xinit, X_reach))  # init & ntasks
H_reach_init = Xlist_to_Hlist(X_reach_init)  # init & ntasks

# taskspace relationship analysis
tspace_mapping = Naive_task_space_correlation(H_reach_init)
# tspace_mapping = KRNN_task_space_correlation(
#     H_reach_init,
#     w_rot=0.0,
#     nnr=0.15,
#     nnk=10,
# )
# Warg = {"wse3_rot": 1.0}
# tspace_mapping = Advanced_task_space_correlation(
#     H_reach_init, Qik_reach_init, Qikstate_reach_init, Warg
# )

task_to_nn_dict, task_to_nn_pair, task_to_nn_pair_len = (
    tspace_mapping["task_to_nn_dict"],
    tspace_mapping["task_to_nn_pair"],
    tspace_mapping["task_to_nn_pair_len"],
)
print(f"==>> task_to_nn_dict: \n{task_to_nn_dict}")
print(f"==>> task_to_nn_pair with {task_to_nn_pair_len} pair: \n{task_to_nn_pair}")


def weighted_nan_euclidean_distances(X, Y=None, w=None):
    """
    Fully vectorized weighted NaN-aware Euclidean distance.

    Parameters
    ----------
    X : (n_samples_X, n_features)
    Y : (n_samples_Y, n_features), optional
    w : (n_features,), optional

    Returns
    -------
    D : (n_samples_X, n_samples_Y)
    """
    X = np.asarray(X, dtype=np.float64)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)

    d = X.shape[1]

    if w is None:
        w = np.ones(d, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)

    total_weight = w.sum()

    # Valid entries
    valid = (~np.isnan(X))[:, None, :] & (~np.isnan(Y))[None, :, :]

    # Replace NaN by zero
    X0 = np.nan_to_num(X)
    Y0 = np.nan_to_num(Y)

    # Pairwise differences
    diff = X0[:, None, :] - Y0[None, :, :]

    # Weighted squared differences
    sqdist = np.sum(w * diff**2 * valid, axis=2)

    # Sum of observed weights
    observed = np.sum(w * valid, axis=2)

    # Normalize exactly like sklearn
    D = np.full_like(sqdist, np.nan)

    mask = observed > 0
    D[mask] = np.sqrt(sqdist[mask] * total_weight / observed[mask])

    return D


def weighted_nan_max_joint_diff_distances(X, Y=None, w=None):
    """
    Pairwise weighted NaN-aware Chebyshev (maximum joint difference) distance.

    d(x, y) = max_i |x_i - y_i| / w_i

    Parameters
    ----------
    X : (n_samples_X, n_features)
    Y : (n_samples_Y, n_features), optional
    w : (n_features,), optional
        Positive scaling factors for each feature.

    Returns
    -------
    D : (n_samples_X, n_samples_Y)
    """
    X = np.asarray(X, dtype=np.float64)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)

    d = X.shape[1]

    if w is None:
        w = np.ones(d, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)

    # Valid dimensions
    valid = (~np.isnan(X))[:, None, :] & (~np.isnan(Y))[None, :, :]

    # Pairwise absolute differences
    diff = np.abs(np.nan_to_num(X)[:, None, :] - np.nan_to_num(Y)[None, :, :])

    # Normalize by weights
    diff = diff / w

    # Ignore invalid dimensions
    diff = np.where(valid, diff, -np.inf)

    # Maximum over features
    D = diff.max(axis=2)

    # If no valid feature exists, return NaN
    D[np.all(~valid, axis=2)] = np.nan

    return D


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

    print(f"---------------------------------------------------------")
    nQredpt = np.sum(Qvalid, axis=1)
    n_selected = np.sum(nQredpt)
    n_total = np.prod(Qvalid.shape)
    print(f"==>> Qfilter R debug Info")
    print(f"==>> selected {n_selected} / {n_total} configurations")
    print(f"==>> selected_rate: {n_selected / n_total}")
    print(f"---------------------------------------------------------")
    return Qvalid


def Qfilter_similarity(Q, q, Qs, thresh):
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
    """
    W = np.array([1, 1, 1, 1, 1, 1])  # weight for each joint
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
    print(f"---------------------------------------------------------")
    nQredpt = np.sum(Qvalid, axis=1)
    n_selected = np.sum(nQredpt)
    n_total = np.prod(Qvalid.shape)
    phi_opt_min = np.nanmin(phi_opt)
    phi_opt_max = np.nanmax(phi_opt)
    print(f"==>> Qfilter similarity debug Info")
    print(f"==>> optimal values: min={phi_opt_min}, max={phi_opt_max}")
    print(f"==>> selected {n_selected} / {n_total} configurations")
    print(f"==>> selected_rate: {n_selected / n_total}")
    print(f"---------------------------------------------------------")
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


Q1red_r = Qfilter_R(Qik_reach_init, qinit, Qs=Qikstate_reach_init, r=2 * np.pi)
Q2red_s = Qfilter_similarity(
    Qik_reach_init, qinit, Qs=Qikstate_reach_init, thresh=0.0001
)
Q3red_nn2c = Qfilter_nn2c(
    Qik_reach_init, Qs=Qikstate_reach_init, tmap=tspace_mapping
)

Q4red_Knn2c = Qfilter_Knn2c(
    Qik_reach_init, Qs=Qikstate_reach_init, k=50, tmap=tspace_mapping
)

Q5red_Dnn2c = Qfilter_Dnn2c(
    Qik_reach_init, Qs=Qikstate_reach_init, d=5, tmap=tspace_mapping
)

# choose filter method
Qreduced = [Q1red_r, Q2red_s, Q3red_nn2c, Q4red_Knn2c, Q5red_Dnn2c][0]
check_number_Q(Qreduced)


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

    # cspace eulidean distance
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols))
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
    check_number_E(Estate)

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

    # cspace weighted euclidean distance
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols))
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = weighted_nan_euclidean_distances(Q[i], Q[j], w=W)

    Estate = np.ones_like(E, dtype=bool)  # True/False mask
    # if Qs is not valid, then E is also invalid
    for idx, (i, j) in enumerate(task_to_nn_pair):
        QIJ = np.outer(Qs[i], Qs[j])  # (num_sols, num_sols)
        Estate[idx] = Estate[idx] & QIJ  # update E state
    check_number_E(Estate)

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

    W = 1 / W  # max joint velocity
    E = np.empty((task_to_nn_pair_len, num_sols, num_sols))
    for idx, (i, j) in enumerate(task_to_nn_pair):
        E[idx] = weighted_nan_max_joint_diff_distances(Q[i], Q[j], w=W)

    Estate = np.ones_like(E, dtype=bool)  # True/False mask
    # if Qs is not valid, then E is also invalid
    for idx, (i, j) in enumerate(task_to_nn_pair):
        QIJ = np.outer(Qs[i], Qs[j])  # (num_sols, num_sols)
        Estate[idx] = Estate[idx] & QIJ  # update E state
    check_number_E(Estate)

    Ewmj = np.where(Estate, E, np.inf)
    return Ewmj


# cmax_d = 2 * np.pi
cmax_d = None  # disable cmax_d filtering
Ecf = Eest_colfree(Qik_reach_init, Qreduced, cmax_d, tspace_mapping)

# Wweu = np.array([1, 1, 1, 1, 1, 1])  # weight for each joint
# Eweu = Eest_weighted_euclidean(Qik_reach_init, Qreduced, Wweu, tspace_mapping)

# Wwmj = np.array([1, 1, 1, 1, 1, 1])  # weight for each joint max velocity
# Ewmj = Eest_weighted_max_joint_diff(Qik_reach_init, Qreduced, Wwmj, tspace_mapping)

# * 3rd STAGE GTSP problem formulation and solving
# write, solve, and read GTSP problem
path = os.path.join(rsrc, "gtsp")
pathprob = os.path.join(rsrc, "gtsp", "new_ur5e_sphere_gtsp.gtsp")
pathsol = os.path.join(rsrc, "gtsp", "new_ur5e_sols")

Ecost = np.where(np.isfinite(Ecf), Ecf, 1000)  # cost for infeasible edges
print(f"==>> Ecost.shape: \n{Ecost.shape}")
print(f"==>> Qreduced.shape: \n{Qreduced.shape}")


Qid_true, Qid_true_cont = write_gtsp_file(
    filename=pathprob,
    instancename="ur5e_sphere_gtsp",
    task_to_nn_pair=task_to_nn_pair,
    E=Ecost,
    Q=Qreduced,
)
result = call_gtsp_glns_solver(
    solver_dir=path,  # Where GLNScmd.jl lives
    input_file=pathprob,
    output_file=pathsol,
    args={"mode": "slow", "max_time": 300},
)
Qtour = read_gtsp_file(pathsol, Qid_true, Qid_true_cont)
Ttour = Qtour // num_sols
print(f"==>> Ttour: \n{Ttour}")

tourQval = Qik_reach_init.reshape(-1, dof)[Qtour]
tourQcosteuldist = np.sum(np.linalg.norm(np.diff(tourQval, axis=0), axis=1))
print(f"==>> tourQcosteuldist: \n{tourQcosteuldist}")


def lininterp_tour(Q, num_points):
    """
    Linear interpolation of the tour path for visualization.
    Q: (n, dof)
    return: (n, num_points, dof)
    """
    Qinterp = np.empty((Q.shape[0] - 1, num_points, Q.shape[1]))
    for i in range(Q.shape[0] - 1):
        Qinterp[i] = np.linspace(Q[i], Q[i + 1], num_points)
    return Qinterp.reshape(-1, Q.shape[1])


Qfull = lininterp_tour(tourQval, num_points=20)


# raise

scene.view_animation(Qfull, Hlist=H)


# * 4th STAGE path reconstruction and refinement
def interp(Q1, Q2, num_points):
    """
    Pairwise linear interpolation between corresponding rows of two sets of configurations.
    Q1: (n, dof)
    Q2: (n, dof)
    num_points: number of interpolation points (including endpoints)
    return: (n, n, num_points, dof)
    """
    # Reshape for broadcasting: (n, 1, 1, dof) and (1, n, 1, dof)
    Q1_exp = Q1[:, np.newaxis, np.newaxis, :]
    Q2_exp = Q2[np.newaxis, :, np.newaxis, :]
    # Interpolation parameter: (1, 1, num_points, 1)
    t = np.linspace(0, 1, num_points)[np.newaxis, np.newaxis, :, np.newaxis]
    # Linear interpolation with full broadcasting
    result = Q1_exp + (Q2_exp - Q1_exp) * t  # (n, n, num_points, dof)
    # Propagate NaN: if Q1 or Q2 has NaN, result is NaN
    result = np.where(np.isnan(Q1_exp) | np.isnan(Q2_exp), np.nan, result)
    return result


def center(Q1, Q2):
    """
    Compute the center configuration between two sets of configurations.
    Q1: (n, dof)
    Q2: (n, dof)
    return: (n, dof)
    """
    # Compute the center while handling NaN values
    Qcenter = np.where(np.isnan(Q1) | np.isnan(Q2), np.nan, (Q1 + Q2) / 2.0)
    return Qcenter


# t1 = 1
# t2 = 2
# idx = task_to_nn_pair.index((t1, t2))
# print(f"==>> idx: \n{idx}")
# Qfrom = Qik_reach_init[t1]  # (num_sols, dof)
# print(f"==>> Qfrom.shape: \n{Qfrom.shape}")
# print(f"==>> Qfrom: \n{Qfrom}")
# Qto = Qik_reach_init[t2]  # (num_sols, dof)
# print(f"==>> Qto.shape: \n{Qto.shape}")
# print(f"==>> Qto: \n{Qto}")
# Qinterp = interp(Qfrom, Qto, num_points=20)
# print(f"==>> Qinterp.shape: \n{Qinterp.shape}")


# # write tour path to file
# pathtour = os.path.join(rsrc, "gtsp", "ur5e_tour_path.txt")
# write_tour_path(pathtour, Qfull)
