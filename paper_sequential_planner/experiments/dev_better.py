import os
import numpy as np
import time
import tqdm
import torch
from paper_sequential_planner.scripts.geometric_torus import (
    find_alt_config2,
    find_altconfig_redudancy,
    find_altconfig_redudancy_fast,
)
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_poses import (
    poses_d,
    poses_c,
    H_to_X,
    xlist_to_Xlist,
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    se3_error_pairwise_distance,
    task_space_correlation,
    task_space_correlation_map,
    filter_cspace_candidate_similar_to_qinit,
    filter_cspace_candidate_radius_to_qinit,
    query_data_from_tspace_map,
)
from paper_sequential_planner.experiments.env_ur5e_sphere import (
    RobotUR5eKin,
    SceneUR5eSpherized,
    device,
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


qinit = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
Xinit = H_to_X(robkin.solve_fk(qinit))
H = poses_d()
X = Hlist_to_Xlist(H)
ntasks = X.shape[0]
Qik = wspace_ik_extended(robkin, X)
Qikstate = wspace_ik_validity_extended(Qik, scene)

Xunreach = np.all(Qikstate == -1, axis=1).flatten()
X_reach = X[~Xunreach]
Qik_reach = Qik[~Xunreach]
Qikstate_reach = Qikstate[~Xunreach]


# taskspace neighborhood analysis
X_reach_init = np.vstack((Xinit, X_reach))  # init & ntasks
tspace_dist = se3_error_pairwise_distance(Xlist_to_Hlist(X_reach_init))
tspace_coorrelation = task_space_correlation(tspace_dist)
tspace_mapping = task_space_correlation_map(tspace_coorrelation)
task_to_nn_mapping, task_to_nn_unique, task_to_nn_unique_len = (
    tspace_mapping["task_to_nn_mapping"],
    tspace_mapping["task_to_nn_unique"],
    tspace_mapping["task_to_nn_unique_len"],
)
print(f"==>> task_to_nn_mapping: \n{task_to_nn_mapping}")
print(f"==>> task_to_nn_unique (i<j): \n{task_to_nn_unique}")
print(f"==>> number of unique undirected edges: \n{task_to_nn_unique_len}")


# cspace
qinit_ = np.full((1, Qik_reach.shape[1], Qik_reach.shape[2]), np.nan)
qinit_[0, 0] = qinit
Qik_reach_init = np.vstack((qinit_, Qik_reach))  # init & ntasks

qinit_state_ = np.full((1, Qikstate_reach.shape[1], Qikstate_reach.shape[2]), -1)
qinit_state_[0, 0] = 1
Qikstate_reach_init = np.vstack((qinit_state_, Qikstate_reach))  # init & ntasks

# constants for GTSP
epslGH = 1.0
eta_collision = 0.1  # resolution of collision checking
cmax_d = 2 * np.pi  # max distance in cspace
r_max = 2 * np.pi  # max radius for candidate selection in cspace
tmult = 0.08  # threshold multiplier for candidate selection in cspace


# eliminate nodes that are too far from qinit in cspace
fr = filter_cspace_candidate_radius_to_qinit(Qik_reach, qinit, radius=r_max)
selected_q, Qin_radius, task_ids_radius, qi_in_task_radius = (
    fr["selected_q"],
    fr["Qin_radius"],
    fr["task_ids"],
    fr["qi_in_task"],
)
print(f"==>> selected_q shape: \n{selected_q.shape}")
# print(f"==>> Qin_radius: \n{Qin_radius}")
# print(f"==>> task_ids_radius: \n{task_ids_radius}")
# print(f"==>> qi_in_task_radius: \n{qi_in_task_radius}")

# check the reachable filter and the cspace radius filter
# this is not include qinit
Q_c = np.where(Qikstate_reach == 1, True, False) & selected_q
qinit_state_selected = np.where(qinit_state_ == 1, True, False)
Q_c_init = np.vstack((qinit_state_selected, Q_c))
num_selected_per_task = np.sum(Q_c_init, axis=1)
print(f"==>> num_selected_per_task: \n{num_selected_per_task}")

# eliminate edges that are too long in cspace
# ! careful with Qik_reach_init, I don't the unreachable to nan yet
# ! must ensure unreachable with Qikstate_react_init
cspace_eudist = np.empty((task_to_nn_unique_len, num_sols, num_sols))
for idx, (i, j) in enumerate(task_to_nn_unique):
    Qt_from = Qik_reach_init[i]  # (num_sols, dof)
    Qt_to = Qik_reach_init[j]  # (num_sols, dof)
    CspaceDistT1T2 = nan_euclidean_distances(Qt_from, Qt_to)
    cspace_eudist[idx] = CspaceDistT1T2

cspace_eudist_state = cspace_eudist <= cmax_d  # True/False
cspace_eudist_val = np.where(cspace_eudist_state, cspace_eudist, np.inf)
# how many edges are valid per task pair
num_valid_edges_per_pair = np.sum(cspace_eudist_state, axis=(1, 2))
print(f"==>> num_valid_edges_per_pair: \n{num_valid_edges_per_pair}")

# eliminate edges more: the one that its node is not selected by the radius filter
cspace_eudist_val_selected = np.empty_like(cspace_eudist_state, dtype=bool)
for idx, (i, j) in enumerate(task_to_nn_unique):
    cspace_eudist_state_ij = cspace_eudist_state[idx]
    Q_c_i = Q_c_init[i]
    Q_c_j = Q_c_init[j]
    Q_c_ij = np.outer(Q_c_i, Q_c_j)  # (num_sols, num_sols)
    cspace_eudist_val_selected[idx] = cspace_eudist_state_ij & Q_c_ij

# check how many edges are left after both filters
num_valid_edges_per_pair_filter = np.sum(cspace_eudist_val_selected, axis=(1, 2))
print(f"==>> num_valid_edges_per_pair_filter: \n{num_valid_edges_per_pair_filter}")


def interp_rack(Q1, Q2, num_points):
    """
    Pairwise linear interpolation between corresponding rows of two sets of configurations.

    Q1: (n, dof)
    Q2: (n, dof)
    num_points: number of interpolation points (including endpoints)

    return: (n, num_points, dof)
    """
    n, dof = Q1.shape

    # Create interpolation parameter: [0, 1] with num_points samples
    t = np.linspace(0, 1, num_points)  # shape: (num_points,)

    # For pairwise interpolation between corresponding rows, broadcast over the time axis:
    # Q1[:, None, :] -> (n, 1, dof)
    # (Q2 - Q1)[:, None, :] -> (n, 1, dof)
    # t[None, :, None] -> (1, num_points, 1)
    t_reshaped = t[None, :, None]  # (1, num_points, 1)
    Q1_reshaped = Q1[:, None, :]  # (n, 1, dof)
    delta = (Q2 - Q1)[:, None, :]  # (n, 1, dof)

    # Linear interpolation: Q(t) = Q1 + t * (Q2 - Q1)
    result = Q1_reshaped + t_reshaped * delta  # (n, num_points, dof)
    return result


# Tid can not be immediately used. must convert back to task indices with task_to_nn_unique
Qfrom = None
Qto = None
nn_id_track = None
nn_count_track = None

for idx, (i, j) in enumerate(task_to_nn_unique):
    cspace_eudist_val_selected_ij = cspace_eudist_val_selected[idx]
    valid_edge_indices = np.where(cspace_eudist_val_selected_ij)
    qfrom_indices = valid_edge_indices[0]
    qto_indices = valid_edge_indices[1]

    qf = Qik_reach_init[i, qfrom_indices]
    qt = Qik_reach_init[j, qto_indices]
    nnid = [idx] * len(qfrom_indices)
    nnc = len(qfrom_indices)
    if Qfrom is None:
        Qfrom = qf
        Qto = qt
        nn_id_track = nnid
        nn_count_track = [nnc]
    else:
        Qfrom = np.vstack((Qfrom, qf))
        Qto = np.vstack((Qto, qt))
        nn_id_track.extend(nnid)
        nn_count_track.append(nnc)


print(f"==>> Qfrom.shape: \n{Qfrom.shape}")
print(f"==>> Qto.shape: \n{Qto.shape}")

# Qcenter = (Qfrom + Qto) / 2.0
# print(f"==>> Qcenter.shape: \n{Qcenter.shape}")


# must compute number of batch, iteration based on pc memory
iter_pair = 10
npair_per_iter = Qfrom.shape[0] // iter_pair
ninterp_points = 20
iter_cc = 20

for pair_idx in tqdm.tqdm(range(iter_pair), desc="pair batches", position=0):
    start = pair_idx * npair_per_iter
    end = (pair_idx + 1) * npair_per_iter
    Qfrom_batch = Qfrom[start:end]
    Qto_batch = Qto[start:end]
    interp = interp_rack(Qfrom_batch, Qto_batch, num_points=ninterp_points)

    # interp now has shape (n_pairs, num_points, dof)
    Q = interp.reshape(-1, interp.shape[2])
    Qtorch = torch.as_tensor(Q, device=device, dtype=torch.float32)

    col_data = torch.empty(Qtorch.shape[0], dtype=torch.bool).to(device)
    nq_per_cc = Qtorch.shape[0] // iter_cc
    for cc_idx in tqdm.tqdm(range(iter_cc), desc="cc", position=1, leave=False):
        start = cc_idx * nq_per_cc
        end = (cc_idx + 1) * nq_per_cc
        Qtorch_batch = Qtorch[start:end]
        col_states = scene.collision_check(Qtorch_batch)
        col_data[start:end] = col_states
        collision_rate = col_data.sum().item() / Qtorch.shape[0]
    # print(f"==>> Dataset collision rate: {collision_rate * 100:.2f}%")
    # break
