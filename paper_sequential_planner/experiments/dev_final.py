import os
import numpy as np
import time
import tqdm
import torch
from paper_sequential_planner.scripts.geometric_torus import (
    find_alt_config2,
)
from sklearn.metrics.pairwise import nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_poses import (
    H_to_X,
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    se3_error_pairwise_distance,
    task_space_correlation,
    task_space_correlation_map,
    query_data_from_tspace_map,
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
qinit = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
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
X_reach_init = np.vstack((Xinit, X_reach))  # init & ntasks

# taskspace distance
tspace_dist = se3_error_pairwise_distance(Xlist_to_Hlist(X_reach_init), w_rot=1.0)
tspace_coorrelation = task_space_correlation(tspace_dist)
tspace_mapping = task_space_correlation_map(tspace_coorrelation)
task_to_nn_dict, task_to_nn_pair, task_to_nn_pair_len = (
    tspace_mapping["task_to_nn_dict"],
    tspace_mapping["task_to_nn_pair"],
    tspace_mapping["task_to_nn_pair_len"],
)
print(f"==>> tspace_dist: \n{tspace_dist}")
print(f"==>> task_to_nn_dict: \n{task_to_nn_dict}")
print(f"==>> task_to_nn_pair with {task_to_nn_pair_len} pair: \n{task_to_nn_pair}")

# cspace distance
Ecspace_eudist = np.empty((task_to_nn_pair_len, num_sols, num_sols))
for idx, (i, j) in enumerate(task_to_nn_pair):
    Qfrom = Qik_reach_init[i]  # (num_sols, dof)
    Qto = Qik_reach_init[j]  # (num_sols, dof)
    Ecspace_eudist[idx] = nan_euclidean_distances(Qfrom, Qto)

# def euclidean_fn(x, y, weights=None):
#   if weights is None:
#     distance = math.sqrt(np.sum((x-y)**2))
#   else:
#     distance = math.sqrt(np.sum(weights*(x-y)**2))
#   return distance
# def max_joint_diff_fn(x, y, weights=None):
#   if weights is None:
#     distance = max(np.abs(x-y))
#   else:
#     distance = max(weights*np.abs(x-y))
#   return distance


def Qfilter_R(Q, qinit, r):
    """
    Filter candidate configurations that are too far from initial configuration
    using a simple radius threshold in cspace distance.
    """
    ntasks_rech, n_ik, dof = Q.shape
    Q_flat = Q.reshape(ntasks_rech * n_ik, dof)
    dist = nan_euclidean_distances(Q_flat, qinit.reshape(1, -1))
    q_valid = dist.flatten() <= r
    q_valid_shape = q_valid.reshape(ntasks_rech, n_ik)
    q_valid_shape = q_valid_shape[:, :, None]  # just add a dummy dimension

    nQredpt = np.sum(q_valid_shape, axis=1)
    n_selected = np.sum(nQredpt)
    n_total = np.prod(q_valid_shape.shape)
    print(f"==>> selected {n_selected} / {n_total} configurations")
    print(f"==>> selected_rate: {n_selected / n_total}")
    return q_valid_shape


def Qfilter_similarity(Q, q, thresh):
    """
    CASE2022 An Efficient Approach for solving RTSP, Li

    Filter candidate configurations that are too far from initial configuration
    Weighted euclidean distance to initial config. Now i dont have the weight yet.
    Bigger val mean closer to qinit mean very little Q selected.
    Samaller val mean farther to qinit mean more Q selected.
    """
    qw = np.array([1, 1, 1, 1, 1, 1])  # weight for each joint
    ntasks_rech, n_ik, dof = Q.shape
    Q_flat = Q.reshape(ntasks_rech * n_ik, dof)
    dist = nan_euclidean_distances(Q_flat, q.reshape(1, -1))
    del_sim = 1.0 / (dist + 0.001)  # avoid division by zero
    phi_opt = del_sim / np.nansum(del_sim)  # normalize to sum to 1
    q_valid = phi_opt >= thresh
    q_valid_shape = q_valid.reshape(ntasks_rech, n_ik)
    q_valid_shape = q_valid_shape[:, :, None]  # just add a dummy dimension

    # threshold = thresh_mult * (optimal_val_max - optimal_val_min) + optimal_val_min

    nQredpt = np.sum(q_valid_shape, axis=1)
    n_selected = np.sum(nQredpt)
    n_total = np.prod(q_valid_shape.shape)
    phi_opt_min = np.nanmin(phi_opt)
    phi_opt_max = np.nanmax(phi_opt)
    print(f"==>> optimal values: min={phi_opt_min}, max={phi_opt_max}")
    print(f"==>> selected {n_selected} / {n_total} configurations")
    print(f"==>> selected_rate: {n_selected / n_total}")
    return q_valid_shape


def Qfilter_nn2c(Q, q):
    pass


# Q filter : Radius filter
Q1red_r = Qfilter_R(Qik_reach_init, qinit, r=2 * np.pi)
check_number_Q(Q1red_r)
Q2red_s = Qfilter_similarity(Qik_reach_init, qinit, thresh=0.0001)
check_number_Q(Q2red_s)

# Merge all Q filter
Qreduced = np.where(Qikstate_reach_init == 1, True, False) & Q1red_r
check_number_Q(Qreduced)

raise


def Efilter_colfree(E, Q, tpair):
    """
    Input:
    E: edges euclidean distance
    Q: nodes validity
    tpair: pairs of tasks tuple

    Output:
    Ecf: edges collision-free distance
    """
    cmax_d = 2 * np.pi
    Estate = E <= cmax_d  # True/False mask dist over cmax_d
    # if Q is not valid, then E is also invalid
    for idx, (i, j) in enumerate(tpair):
        QIJ = np.outer(Q[i], Q[j])  # (num_sols, num_sols)
        Estate[idx] = Estate[idx] & QIJ  # update E state

    check_number_E(Estate)
    # from the E state, we estimation the collision-free distance
    # If Estate is True then we have distance valid
    # If Estate is False then we have distance as np.inf
    # For now we use fake cost
    Ecf = np.where(Estate, E, np.inf) + np.random.random(E.shape) * 1e-6
    return Ecf


Ecf = Efilter_colfree(Ecspace_eudist, Qreduced, task_to_nn_pair)

raise
# * 3rd STAGE GTSP problem formulation and solving
# write, solve, and read GTSP problem
path = os.path.join(rsrc, "gtsp")
pathprob = os.path.join(rsrc, "gtsp", "new_ur5e_sphere_gtsp.gtsp")
pathsol = os.path.join(rsrc, "gtsp", "new_ur5e_sols")

Ecost = np.where(np.isfinite(Ecf), Ecf, 1000)  # cost for infeasible edges
Qreduced_final_flat, nodesid_og, nodesid_cont = write_gtsp_file(
    filename=pathprob,
    name="ur5e_sphere_gtsp",
    task_to_nn_pair=task_to_nn_pair,
    Ecost=Ecost,
    Q=Qreduced,
)
result = call_gtsp_glns_solver(
    solver_dir=path,  # Where GLNScmd.jl lives
    input_file=pathprob,
    output_file=pathsol,
    args={"mode": "slow", "max_time": 300},
)
tour_flatten = read_gtsp_file(pathsol, nodesid_og, nodesid_cont)
print(f"==>> tour_flatten: \n{tour_flatten}")
Qik_reach_init_flat = Qik_reach_init.reshape(-1, dof)
qtour = Qik_reach_init_flat[tour_flatten]
print(f"==>> qtour.shape: \n{qtour.shape}")
print(f"==>> qtour: \n{qtour}")

tour_euc_cost = np.sum(np.linalg.norm(np.diff(qtour, axis=0), axis=1))
print(f"==>> tour_euc_cost: \n{tour_euc_cost}")
raise


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


t1 = 1
t2 = 2
idx = task_to_nn_pair.index((t1, t2))
print(f"==>> idx: \n{idx}")
Qfrom = Qik_reach_init[t1]  # (num_sols, dof)
print(f"==>> Qfrom.shape: \n{Qfrom.shape}")
print(f"==>> Qfrom: \n{Qfrom}")
Qto = Qik_reach_init[t2]  # (num_sols, dof)
print(f"==>> Qto.shape: \n{Qto.shape}")
print(f"==>> Qto: \n{Qto}")
Qinterp = interp(Qfrom, Qto, num_points=20)
print(f"==>> Qinterp.shape: \n{Qinterp.shape}")


raise
# write tour path to file
pathtour = os.path.join(rsrc, "gtsp", "ur5e_tour_path.txt")
write_tour_path(pathtour, qtour)
