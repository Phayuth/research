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

from paper_sequential_planner.experiments.utilio import write_gtsp_file

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
H = poses_d()
X = Hlist_to_Xlist(H)
ntasks = X.shape[0]
Qik = wspace_ik_extended(robkin, X)
Qikstate = wspace_ik_validity_extended(Qik, scene)

# filter out the unreachable tasks
Xunreach = np.all(Qikstate == -1, axis=1).flatten()
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

# ----------------- All method must start from here -------------------------------

# taskspace neighborhood analysis
tspace_dist = se3_error_pairwise_distance(Xlist_to_Hlist(X_reach_init))
tspace_coorrelation = task_space_correlation(tspace_dist)
tspace_mapping = task_space_correlation_map(tspace_coorrelation)
task_to_nn_dict, task_to_nn_pair, task_to_nn_pair_len = (
    tspace_mapping["task_to_nn_dict"],
    tspace_mapping["task_to_nn_pair"],
    tspace_mapping["task_to_nn_pair_len"],
)
print(f"==>> task_to_nn_dict: \n{task_to_nn_dict}")
print(f"==>> task_to_nn_pair with {task_to_nn_pair_len} pair: \n{task_to_nn_pair}")


# constants for GTSP
epslGH = 1.0
eta_collision = 0.1  # resolution of collision checking
cmax_d = 2 * np.pi  # max distance in cspace
r_max = 2 * np.pi  # max radius for candidate selection in cspace
tmult = 0.08  # threshold multiplier for candidate selection in cspace


# eliminate nodes that are too far from qinit in cspace
Qreduced = filter_cspace_candidate_radius_to_qinit(Qik_reach, qinit, radius=r_max)
print(f"==>> Qreduced shape: \n{Qreduced.shape}")

# check the reachable filter and the cspace radius filter
Qred_ = np.where(Qikstate_reach == 1, True, False) & Qreduced
qinit_red_ = np.where(qinit_state_ == 1, True, False)
Qreduced_init = np.vstack((qinit_red_, Qred_))
nQredpt = np.sum(Qreduced_init, axis=1)
print(f"==>> nQredpt: \n{nQredpt.T}")

# determine cspace euclidean distance between all the valid edges in taskspace
Ecspace_eudist = np.empty((task_to_nn_pair_len, num_sols, num_sols))
for idx, (i, j) in enumerate(task_to_nn_pair):
    Qfrom = Qik_reach_init[i]  # (num_sols, dof)
    Qto = Qik_reach_init[j]  # (num_sols, dof)
    Ecspace_eudist[idx] = nan_euclidean_distances(Qfrom, Qto)

# eliminate edges that are too long in cspace
# ! careful with Qik_reach_init, I don't check the unreachable to nan yet
# ! must ensure unreachable with Qikstate_react_init
Ecspace_eudist_state = Ecspace_eudist <= cmax_d  # True/False mask dist over cmax_d
nEpt = np.sum(Ecspace_eudist_state, axis=(1, 2))
print(f"==>> nEpt: \n{nEpt}")

# eliminate edges more: the one that its node is not selected by the radius filter
Ecspace_eudist_red = np.empty_like(Ecspace_eudist_state, dtype=bool)
for idx, (i, j) in enumerate(task_to_nn_pair):
    Ecspace_eudist_state_ij = Ecspace_eudist_state[idx]
    Qreduced_i = Qreduced_init[i]
    Qreduced_j = Qreduced_init[j]
    Qreduced_ij = np.outer(Qreduced_i, Qreduced_j)  # (num_sols, num_sols)
    Ecspace_eudist_red[idx] = Ecspace_eudist_state_ij & Qreduced_ij

nEvalpt = np.sum(Ecspace_eudist_red, axis=(1, 2))
print(f"==>> nEvalpt: \n{nEvalpt}")
nEval = np.sum(nEvalpt)
print(f"==>> nEval: \n{nEval}")


# 2nd round of Q filtered after the edge elimination
Qreduced_final = np.full_like(Qreduced_init, False, dtype=bool)
for idx, (i, j) in enumerate(task_to_nn_pair):
    Ecspace_eudist_red_ij = Ecspace_eudist_red[idx]
    q_from = np.any(Ecspace_eudist_red_ij, axis=1)[..., np.newaxis]
    q_to = np.any(Ecspace_eudist_red_ij, axis=0)[..., np.newaxis]
    Qreduced_final[i] = np.logical_or(Qreduced_final[i], q_from)
    Qreduced_final[j] = np.logical_or(Qreduced_final[j], q_to)
print(f"==>> Qreduced_final.shape: \n{Qreduced_final.shape}")

nQredfinalpt = np.sum(Qreduced_final, axis=1)
print(f"==>> nQredfinalpt: \n{nQredfinalpt.T}")
nQredfinal = np.sum(nQredfinalpt)
print(f"==>> nQredfinal: \n{nQredfinal}")


# mapping the flatten node id
# nodeid_og is the original node id
# nodeid_cont is the continuous node id for gtsp solver
Qreduced_final_flat = Qreduced_final.flatten()
nodesid_og = np.where(Qreduced_final_flat)[0]  # take only the True nodes
nodesid_cont = np.arange(nQredfinal) + 1  # GTSP node id start from 1
print(f"==>> nodesid_og.shape: \n{nodesid_og.shape}")
print(f"==>> nodesid_og: \n{nodesid_og}")
print(f"==>> nodesid_cont.shape: \n{nodesid_cont.shape}")
print(f"==>> nodesid_cont: \n{nodesid_cont}")


# def generate_gtsp_header(name, dimension, gtsp_sets):
#     """
#     DIMENSION is the total sum of every nodes in every cluster (solutions)
#     GTSP_SETS is the total number of clusters (tasks)
#     """
#     lines = []
#     lines.append(f"NAME: {name}")
#     lines.append("TYPE: GTSP")
#     lines.append(f"DIMENSION: {dimension}")
#     lines.append(f"GTSP_SETS: {gtsp_sets}")
#     lines.append("EDGE_WEIGHT_TYPE: EXPLICIT")
#     lines.append("EDGE_WEIGHT_FORMAT: FULL_MATRIX")
#     return "\n".join(lines)


# def generate_gtsp_edge_weight_section():
#     # Extract task and solution IDs for each flat node
#     node_tasks = nodesid_og // num_sols
#     node_sols = nodesid_og % num_sols
#     n_nodes = len(nodesid_og)

#     # Build lookup table for task pair indices: task_pair_lookup[i,j] -> task_pair_idx
#     task_to_nn_pair_arr = np.array(task_to_nn_pair)
#     num_tasks = task_to_nn_pair_arr.max() + 1
#     task_pair_lookup = np.full((num_tasks, num_tasks), -1, dtype=np.int32)
#     for idx, (i, j) in enumerate(task_to_nn_pair_arr):
#         task_pair_lookup[i, j] = idx
#         task_pair_lookup[j, i] = idx

#     # Create meshgrid for all node pairs
#     i_idx, j_idx = np.meshgrid(
#         np.arange(n_nodes), np.arange(n_nodes), indexing="ij"
#     )
#     i_idx_flat = i_idx.ravel()
#     j_idx_flat = j_idx.ravel()

#     # Get task/sol for each node in all pairs
#     task_i = node_tasks[i_idx_flat]
#     sol_i = node_sols[i_idx_flat]
#     task_j = node_tasks[j_idx_flat]
#     sol_j = node_sols[j_idx_flat]

#     # Initialize distance matrix
#     gtsp_dist_matrix = np.full((n_nodes, n_nodes), 1000, dtype=np.float64)
#     np.fill_diagonal(gtsp_dist_matrix, 0)  # Set diagonal to 0

#     # Find task pair indices using lookup table
#     task_pair_idx = task_pair_lookup[task_i, task_j]

#     # Valid pairs: same task pair exists and tasks are different
#     valid_pairs = (task_pair_idx >= 0) & (task_i != task_j)

#     if np.any(valid_pairs):
#         task_pair_idx_valid = task_pair_idx[valid_pairs]
#         i_idx_valid = i_idx_flat[valid_pairs]
#         j_idx_valid = j_idx_flat[valid_pairs]
#         task_i_valid = task_i[valid_pairs]
#         sol_i_valid = sol_i[valid_pairs]
#         task_j_valid = task_j[valid_pairs]
#         sol_j_valid = sol_j[valid_pairs]

#         # For each valid pair, check if edge is valid in cspace_eudist_val_selected
#         for idx in range(len(task_pair_idx_valid)):
#             tp_idx = task_pair_idx_valid[idx]
#             ti = task_i_valid[idx]
#             tj = task_j_valid[idx]
#             si = sol_i_valid[idx]
#             sj = sol_j_valid[idx]
#             i_pos = i_idx_valid[idx]
#             j_pos = j_idx_valid[idx]

#             if ti < tj:
#                 is_valid = Ecspace_eudist_red[tp_idx, si, sj]
#                 if is_valid:
#                     gtsp_dist_matrix[i_pos, j_pos] = Ecspace_eudist[tp_idx, si, sj]
#             else:
#                 is_valid = Ecspace_eudist_red[tp_idx, sj, si]
#                 if is_valid:
#                     gtsp_dist_matrix[i_pos, j_pos] = Ecspace_eudist[tp_idx, sj, si]

#     gtsp_dist_matrix_int = (gtsp_dist_matrix * 1000).astype(int)

#     # Generate GTSP_EDGE_WEIGHT_SECTION
#     lines = ["EDGE_WEIGHT_SECTION"]
#     for i in range(n_nodes):
#         row = gtsp_dist_matrix_int[i]
#         row_str = " ".join(str(int(x)) for x in row)
#         lines.append(row_str)

#     return "\n".join(lines)


# def generate_gtsp_set_section():
#     lines = ["GTSP_SET_SECTION"]
#     node_idx = 0
#     for task_id, num_nodes in enumerate(nQredfinalpt, start=1):
#         # Get the nodes for this task
#         task_nodes = nodesid_cont[node_idx : node_idx + num_nodes.item()]
#         # Format: task_id node1 node2 ... nodeN -1
#         nodes_str = " ".join(map(str, task_nodes))
#         lines.append(f"{task_id} {nodes_str} -1")
#         node_idx += num_nodes.item()
#     lines.append("EOF")
#     return "\n".join(lines)


# def write_gtsp_file(filename="output.gtsp"):
#     header = generate_gtsp_header()
#     edge_section = generate_gtsp_edge_weight_section()
#     set_section = generate_gtsp_set_section()

#     # Concatenate all sections
#     gtsp_content = f"{header}\n\n{edge_section}\n\n{set_section}"

#     # Write to file
#     with open(filename, "w") as f:
#         f.write(gtsp_content)

#     print(f"GTSP file written to {filename}")

write_gtsp_file(
    filename="ur5e_sphere_gtsp.gtsp",
    name="ur5e_sphere_gtsp",
    task_to_nn_pair=task_to_nn_pair,
    Ecspace_eudist=Ecspace_eudist,
    Ecspace_eudist_red=Ecspace_eudist_red,
    Ecspace_colfree=None,
    Qreduced_final=Qreduced_final,
)
# write_gtsp_file("ur5e_sphere_gtsp.gtsp")
tour = [
    302,
    1,
    367,
    341,
    317,
    29,
    45,
    63,
    114,
    106,
    23,
    89,
    102,
    221,
    280,
    268,
    191,
    164,
    255,
    244,
    135,
    405,
    398,
    392,
    384,
]

# remap the tour flatten node id back to its original id
tour_indices = np.searchsorted(nodesid_cont, tour)
print(f"==>> tour_indices: \n{tour_indices}")
tour_indices_og = nodesid_og[tour_indices]
print(f"==>> tour_indices_og: \n{tour_indices_og}")

raise


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
    Ecspace_eudist_red_ij = Ecspace_eudist_red[idx]
    valid_edge_indices = np.where(Ecspace_eudist_red_ij)
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


raise
print(f"==>> Qfrom.shape: \n{Qfrom.shape}")
print(f"==>> Qto.shape: \n{Qto.shape}")
print(f"==>> nn_id_track: \n{nn_id_track}")
print(f"==>> nn_count_track: \n{nn_count_track}")

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


# now assume we have all the valid path and the cost after searching.
# we must remap them back to the task indices
valid_after_search = np.array([True] * Qfrom.shape[0], dtype=bool)
cost_after_search = np.linalg.norm(Qto - Qfrom, axis=1)

Ecspace_eudist_val = np.where(Ecspace_eudist_state, Ecspace_eudist, np.inf)

# write to GTSP format
