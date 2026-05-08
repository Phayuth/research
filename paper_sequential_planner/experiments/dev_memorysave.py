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


# initial config and task  -----------------------------------------
qinit = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
Xinit = H_to_X(robkin.solve_fk(qinit))
H = poses_d()
X = Hlist_to_Xlist(H)
Qaik = wspace_ik_extended(robkin, X)  # (ntasks, 7)
Qaik_valid = wspace_ik_validity_extended(Qaik, scene)  # (ntasks, 7, 1)
X_isunr = np.all(Qaik_valid != 1, axis=1).flatten()  # (ntasks, ) True/False
X_r = X[~X_isunr]  # (ntasks_rech, 2)
Qaik_valid_r = Qaik_valid[~X_isunr]  # (ntasks_rech, n_ik * altcnf, 1)
Qaik_r = Qaik[~X_isunr]  # (ntasks_rech, n_ik * altcnf, dof)
Qaik_r = np.where(Qaik_valid_r == 1, Qaik_r, np.nan)  # set value to nan if invalid
X_rall = np.vstack((Xinit, X_r))  # (ntasks+1, 2) & init
qinit_all = np.full((1, Qaik_r.shape[1], Qaik_r.shape[2]), np.nan)
qinit_all[0, 0] = qinit  # set first IK to qinit and rest to nan, real val no alt
Qaik_rall = np.vstack((qinit_all, Qaik_r))  # (ntasks+1, n_ik*altcnf, dof) & init
qinit_valid = np.full((1, Qaik_valid_r.shape[1], Qaik_valid_r.shape[2]), -1)
qinit_valid[0, 0] = 1  # set first IK valid to qinit and rest to -1 (no sol)
Qaik_valid_rall = np.vstack((qinit_valid, Qaik_valid_r))
print(f"==>> X_rall.shape: \n{X_rall.shape}")
print(f"==>> Qaik_rall.shape: \n{Qaik_rall.shape}")
print(f"==>> Qaik_valid_rall.shape: \n{Qaik_valid_rall.shape}")
# -----------------------------------------------------------------

# taskspace distance -----------------------------------------------
H_r_full = Xlist_to_Hlist(X_rall)  # (ntasks_rech, 4, 4)
tspace_dist = se3_error_pairwise_distance(H_r_full, 0.2)  # (ntasks_r, ntasks_r)
print(f"==>> tspace_dist.shape: \n{tspace_dist.shape}")

# tspace neighborhood and mapping --------------------------------------
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

ntasks_rech, _, _ = Qaik_rall.shape
print(f"==>> ntasks_rech: \n{ntasks_rech}")

# compute cspace distance for all unique task pairs (i, j) where i < j
cspace_eudist = np.empty((task_to_nn_unique_len, num_sols, num_sols))
for idx, (i, j) in enumerate(task_to_nn_unique):
    Qt1 = Qaik_rall[i]  # (n_ik*altcnf, dof)
    Qt2 = Qaik_rall[j]  # (n_ik*altcnf, dof)
    CspaceDistT1T2 = nan_euclidean_distances(Qt1, Qt2)
    cspace_eudist[idx] = CspaceDistT1T2
print(f"==>> cspace_eudist.shape: \n{cspace_eudist.shape}")

# compute grouping matrix for all unique task pairs (i, j) where i < j
K_encode = 10  # encode grouping
cspace_group = np.empty_like(cspace_eudist, dtype=np.int32)
for idx, (ti, tj) in enumerate(task_to_nn_unique):
    if ti == 0:
        pass  # ingore task 0 which is qinit for now.
    else:
        for ik_i in range(unique_sols):
            for ik_j in range(unique_sols):
                i0 = ik_i * alt_num
                j0 = ik_j * alt_num
                i1 = i0 + alt_num
                j1 = j0 + alt_num
                Qt1 = Qaik_rall[ti, i0:i1]  # (altcnf, dof)
                Qt2 = Qaik_rall[tj, j0:j1]  # (altcnf, dof)
                group_mat = find_altconfig_redudancy_fast(Qt1, Qt2)[3]
                id_encode = i0 * K_encode + j0
                cspace_group[idx, i0:i1, j0:j1] = group_mat * id_encode


# constants for GTSP
epslGH = 1.0
eta_collision = 0.1  # resolution of collision checking
cmax_d = 2 * np.pi  # max distance in cspace
r_max = 2 * np.pi  # max radius for candidate selection in cspace
tmult = 0.08  # threshold multiplier for candidate selection in cspace

# filter candidate
# eliminate edges
cspace_eudist_v = cspace_eudist <= cmax_d
cspace_eudist_f = np.where(cspace_eudist_v, cspace_eudist, np.nan)
print(f"==>> cspace_eudist_f.shape: \n{cspace_eudist_f.shape}")

# eliminate nodes
fd_sim = filter_cspace_candidate_similar_to_qinit(Qaik_r, qinit, thresh_mult=tmult)
fd_r = filter_cspace_candidate_radius_to_qinit(Qaik_r, qinit, radius=r_max)
Qin_sim, task_ids_sim, qi_in_task_sim = (
    fd_sim["Qin_sim"],
    fd_sim["task_ids"],
    fd_sim["qi_in_task"],
)
Qin_radius, task_ids_radius, qi_in_task_radius = (
    fd_r["Qin_radius"],
    fd_r["task_ids"],
    fd_r["qi_in_task"],
)
print(f"==>> Qin_radius: \n{Qin_radius}")
print(f"==>> task_ids_radius: \n{task_ids_radius}")
print(f"==>> qi_in_task_radius: \n{qi_in_task_radius}")


def interp_rack(Q1, Q2, num_points):
    """
    Pairwise linear interpolation between two sets of configurations.

    Q1: n, dof
    Q2: n, dof
    num_points: number of interpolation points (including endpoints)

    return: n, n, num_points, dof
    """
    n, dof = Q1.shape

    # Create interpolation parameter: [0, 1] with num_points samples
    t = np.linspace(0, 1, num_points)  # shape: (num_points,)

    # Reshape for broadcasting
    Q1_reshaped = Q1[:, np.newaxis, np.newaxis, :]  # (n, 1, 1, dof)
    Q2_reshaped = Q2[np.newaxis, :, np.newaxis, :]  # (1, n, 1, dof)
    t_reshaped = t[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, num_points, 1)

    # Linear interpolation: Q(t) = Q1 + t * (Q2 - Q1)
    result = Q1_reshaped + t_reshaped * (Q2_reshaped - Q1_reshaped)
    return result  # (n, n, num_points, dof)


if __name__ == "__main__":
    I = 1
    J = 5

    Qaik_rall_I = Qaik_rall[I]  # (n_ik*altcnf, dof)
    print(f"==>> Qaik_rall_I: \n{Qaik_rall_I}")
    Qaik_rall_J = Qaik_rall[J]  # (n_ik*altcnf, dof)
    print(f"==>> Qaik_rall_J: \n{Qaik_rall_J}")

    t1qid_select = qi_in_task_radius[task_ids_radius == I]
    print(f"==>> t1qid_select: \n{t1qid_select}")
    t2qid_select = qi_in_task_radius[task_ids_radius == J]
    print(f"==>> t2qid_select: \n{t2qid_select}")

    edge_ck = query_data_from_tspace_map(I, J, cspace_eudist_v, task_to_nn_unique)
    print(f"==>> edge_ck: \n{edge_ck}")

    # get the edges that is true in edge_ck AND the node indices exist in the radius-based candidate selection
    # Ic, Jc = np.where(edge_ck)
    # print(f"==>> Ic: \n{Ic}")
    # print(f"==>> Jc: \n{Jc}")

    # Create 2D boolean masks for valid nodes
    valid_nodes_i = np.isin(np.arange(edge_ck.shape[0]), t1qid_select)
    valid_nodes_j = np.isin(np.arange(edge_ck.shape[1]), t2qid_select)

    # Broadcast to 2D: (n, 1) & (1, m) → (n, m)
    valid_node_pairs = valid_nodes_i[:, np.newaxis] & valid_nodes_j[np.newaxis, :]

    # Combine with edge_ck
    edges_to_check = edge_ck & valid_node_pairs
    Ic, Jc = np.where(edges_to_check)

    print(f"==>> Ic: \n{Ic}")
    print(f"==>> Jc: \n{Jc}")

    group_ckIJ = query_data_from_tspace_map(I, J, cspace_group, task_to_nn_unique)

    gck_I = group_ckIJ[Ic][:, Jc]
    print(f"==>> gck_I: \n{gck_I}")

    Uval, Ucnt = np.unique(gck_I, return_counts=True)
    print(f"==>> Uval: \n{Uval}")
    print(f"==>> Ucnt: \n{Ucnt}")
    # uni_id = len(Ucnt)
    # print(f"==>> uni_id: \n{uni_id}")
    raise
    numpoints = 20
    interp = interp_rack(Qaik_rall_I, Qaik_rall_J, num_points=numpoints)
    print(f"==>> interp.shape: \n{interp.shape}")

    Q = interp.reshape(-1, interp.shape[3])
    Qtorch = torch.as_tensor(Q, device=device, dtype=torch.float32)
    print(f"==>> Qtorch.shape: \n{Qtorch.shape}")
    nbatch = Qtorch.shape[0] // numpoints
    print(f"==>> nbatch: \n{nbatch}")

    col_data = torch.empty(Qtorch.shape[0], dtype=torch.bool).to(device)
    it = Qtorch.shape[0] // nbatch
    print(f"==>> it: \n{it}")
    for i in tqdm.tqdm(range(it)):
        start = i * nbatch
        end = (i + 1) * nbatch
        Qbatch = Qtorch[start:end]
        col_states = scene.collision_check(Qbatch)
        col_data[start:end] = col_states
    collision_rate = col_data.sum().item() / Qtorch.shape[0]
    print(f"==>> Dataset collision rate: {collision_rate * 100:.2f}%")
