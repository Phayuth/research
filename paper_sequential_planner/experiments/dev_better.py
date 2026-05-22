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

# from utilio import None

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
q_in_R = filter_cspace_candidate_radius_to_qinit(Qik_reach, qinit, radius=r_max)
print(f"==>> q_in_R shape: \n{q_in_R.shape}")

# check the reachable filter and the cspace radius filter
Q_c = np.where(Qikstate_reach == 1, True, False) & q_in_R
qinit_state_selected = np.where(qinit_state_ == 1, True, False)
Q_c_init = np.vstack((qinit_state_selected, Q_c))
print(f"==>> Q_c_init.shape: \n{Q_c_init.shape}")

num_selected_per_task = np.sum(Q_c_init, axis=1)
print(f"==>> num_selected_per_task: \n{num_selected_per_task.T}")

# eliminate edges that are too long in cspace
# ! careful with Qik_reach_init, I don't the unreachable to nan yet
# ! must ensure unreachable with Qikstate_react_init
cspace_eudist = np.empty((task_to_nn_unique_len, num_sols, num_sols))
for idx, (i, j) in enumerate(task_to_nn_unique):
    Qt_from = Qik_reach_init[i]  # (num_sols, dof)
    Qt_to = Qik_reach_init[j]  # (num_sols, dof)
    CspaceDistT1T2 = nan_euclidean_distances(Qt_from, Qt_to)
    cspace_eudist[idx] = CspaceDistT1T2

cspace_eudist_state = cspace_eudist <= cmax_d  # True/False mask dist over cmax_d
# Print debug ----------- how many edges are valid per task pair
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

# Print debug ----------- check how many edges are left after both filters
num_valid_edges_per_pair_filter = np.sum(cspace_eudist_val_selected, axis=(1, 2))
print(f"==>> num_valid_edges_per_pair_filter: \n{num_valid_edges_per_pair_filter}")
num_valid_edges_per_pair_filter_total = np.sum(num_valid_edges_per_pair_filter)
print(
    f"==>> total number of valid edges after filter: \n{num_valid_edges_per_pair_filter_total}"
)

# USE this variable for all the downstream processing
# cspace_eudist_val_selected
# print("".center(50, "-"))
# print(cspace_eudist_val_selected)


Q_c_init_exist_after_edges = np.full_like(Q_c_init, False, dtype=bool)
for idx, (i, j) in enumerate(task_to_nn_unique):
    cspace_eudist_val_selected_ij = cspace_eudist_val_selected[idx]
    q_from = np.any(cspace_eudist_val_selected_ij, axis=1)[..., np.newaxis]
    q_to = np.any(cspace_eudist_val_selected_ij, axis=0)[..., np.newaxis]
    Q_c_init_exist_after_edges[i] = np.logical_or(
        Q_c_init_exist_after_edges[i], q_from
    )
    Q_c_init_exist_after_edges[j] = np.logical_or(
        Q_c_init_exist_after_edges[j], q_to
    )
print(
    f"==>> Q_c_init_exist_after_edges.shape: \n{Q_c_init_exist_after_edges.shape}"
)

for taski in range(Q_c_init.shape[0]):
    qs = Q_c_init_exist_after_edges[taski]
    num_qs = np.sum(qs)

total_q_selected = np.sum(Q_c_init_exist_after_edges)
print(f"Total number of configurations selected after: {total_q_selected}")

Q_c_init_exist_after_edges_flat = Q_c_init_exist_after_edges.flatten()
nodesid_true = np.where(Q_c_init_exist_after_edges_flat)[0]
print(f"==>> nodesid_true.shape: \n{nodesid_true.shape}")
print(f"==>> nodesid_true: \n{nodesid_true}")

ntasksss = Q_c_init_exist_after_edges.shape[0]
print(f"==>> ntasksss: \n{ntasksss}")
nodesid = np.arange(total_q_selected) + 1  # GTSP node id start from 1
print(f"==>> nodesid.shape: \n{nodesid.shape}")
print(f"==>> nodesid: \n{nodesid}")


def generate_gtsp_header():
    lines = []
    lines.append("NAME: ur5e_sphere")
    lines.append("TYPE: GTSP")
    lines.append(f"DIMENSION: {total_q_selected}")
    lines.append("GTSP_SETS: {}".format(Q_c_init_exist_after_edges.shape[0]))
    lines.append("EDGE_WEIGHT_TYPE: EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT: FULL_MATRIX")
    return "\n".join(lines)


def generate_gtsp_edge_weight_section():
    # Extract task and solution IDs for each flat node
    node_tasks = nodesid_true // num_sols
    node_sols = nodesid_true % num_sols
    n_nodes = len(nodesid_true)

    # Build lookup table for task pair indices: task_pair_lookup[i,j] -> task_pair_idx
    task_to_nn_unique_arr = np.array(task_to_nn_unique)
    num_tasks = task_to_nn_unique_arr.max() + 1
    task_pair_lookup = np.full((num_tasks, num_tasks), -1, dtype=np.int32)
    for idx, (i, j) in enumerate(task_to_nn_unique_arr):
        task_pair_lookup[i, j] = idx
        task_pair_lookup[j, i] = idx

    # Create meshgrid for all node pairs
    i_idx, j_idx = np.meshgrid(
        np.arange(n_nodes), np.arange(n_nodes), indexing="ij"
    )
    i_idx_flat = i_idx.ravel()
    j_idx_flat = j_idx.ravel()

    # Get task/sol for each node in all pairs
    task_i = node_tasks[i_idx_flat]
    sol_i = node_sols[i_idx_flat]
    task_j = node_tasks[j_idx_flat]
    sol_j = node_sols[j_idx_flat]

    # Initialize distance matrix
    gtsp_dist_matrix = np.full((n_nodes, n_nodes), 1000, dtype=np.float64)
    np.fill_diagonal(gtsp_dist_matrix, 0)  # Set diagonal to 0

    # Find task pair indices using lookup table
    task_pair_idx = task_pair_lookup[task_i, task_j]

    # Valid pairs: same task pair exists and tasks are different
    valid_pairs = (task_pair_idx >= 0) & (task_i != task_j)

    if np.any(valid_pairs):
        task_pair_idx_valid = task_pair_idx[valid_pairs]
        i_idx_valid = i_idx_flat[valid_pairs]
        j_idx_valid = j_idx_flat[valid_pairs]
        task_i_valid = task_i[valid_pairs]
        sol_i_valid = sol_i[valid_pairs]
        task_j_valid = task_j[valid_pairs]
        sol_j_valid = sol_j[valid_pairs]

        # For each valid pair, check if edge is valid in cspace_eudist_val_selected
        for idx in range(len(task_pair_idx_valid)):
            tp_idx = task_pair_idx_valid[idx]
            ti = task_i_valid[idx]
            tj = task_j_valid[idx]
            si = sol_i_valid[idx]
            sj = sol_j_valid[idx]
            i_pos = i_idx_valid[idx]
            j_pos = j_idx_valid[idx]

            if ti < tj:
                is_valid = cspace_eudist_val_selected[tp_idx, si, sj]
                if is_valid:
                    gtsp_dist_matrix[i_pos, j_pos] = cspace_eudist[tp_idx, si, sj]
            else:
                is_valid = cspace_eudist_val_selected[tp_idx, sj, si]
                if is_valid:
                    gtsp_dist_matrix[i_pos, j_pos] = cspace_eudist[tp_idx, sj, si]

    gtsp_dist_matrix_int = (gtsp_dist_matrix * 1000).astype(int)

    # Generate GTSP_EDGE_WEIGHT_SECTION
    lines = ["EDGE_WEIGHT_SECTION"]
    for i in range(n_nodes):
        row = gtsp_dist_matrix_int[i]
        row_str = " ".join(str(int(x)) for x in row)
        lines.append(row_str)

    return "\n".join(lines)


def generate_gtsp_set_section():
    lines = ["GTSP_SET_SECTION"]
    node_idx = 0
    for task_id, num_nodes in enumerate(num_selected_per_task, start=1):
        # Get the nodes for this task
        task_nodes = nodesid[node_idx : node_idx + num_nodes.item()]
        # Format: task_id node1 node2 ... nodeN -1
        nodes_str = " ".join(map(str, task_nodes))
        lines.append(f"{task_id} {nodes_str} -1")
        node_idx += num_nodes.item()
    lines.append("EOF")
    return "\n".join(lines)


def write_gtsp_file(filename="output.gtsp"):
    header = generate_gtsp_header()
    edge_section = generate_gtsp_edge_weight_section()
    set_section = generate_gtsp_set_section()

    # Concatenate all sections
    gtsp_content = f"{header}\n\n{edge_section}\n\n{set_section}"

    # Write to file
    with open(filename, "w") as f:
        f.write(gtsp_content)

    print(f"GTSP file written to {filename}")


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
tour_indices = np.searchsorted(nodesid, tour)
print(f"==>> tour_indices: \n{tour_indices}")
tour_indices_og = nodesid_true[tour_indices]
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

cspace_eudist_val = np.where(cspace_eudist_state, cspace_eudist, np.inf)

# write to GTSP format
