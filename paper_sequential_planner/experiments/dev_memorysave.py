import os
import numpy as np
from paper_sequential_planner.scripts.geometric_torus import (
    find_alt_config2,
    find_altconfig_redudancy,
    find_altconfig_redudancy_fast,
)
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_poses import (
    poses_d,
    H_to_X,
    xlist_to_Xlist,
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    se3_error_pairwise_distance,
    task_space_correlation,
    task_space_correlation_map,
    filter_cspace_candidate_similar_to_qinit,
    filter_cspace_candidate_radius_to_qinit,
    query_tasks_data_from_tspace_map,
)
from paper_sequential_planner.experiments.env_ur5e import RobotUR5eKin

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robkin = RobotUR5eKin()
scene = None  # for collision checking, not implemented yet
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
                isCollsion = False  # robscene.check_collision(q)
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
Qaik_valid = wspace_ik_validity_extended(Qaik, None)  # (ntasks, 7, 1)
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

# # Save validity table as CSV with shape (256, 25): rows=IK configs, cols=tasks.
# qaik_valid_csv = Qaik_valid_rall[:, :, 0].T
# csv_path = os.path.join(os.path.dirname(__file__), "Qaik_valid_rall_256x25.csv")
# np.savetxt(csv_path, qaik_valid_csv, delimiter=",", fmt="%d")
# print(f"==>> saved Qaik_valid_rall CSV: {csv_path}")
# -----------------------------------------------------------------

# taskspace distance -----------------------------------------------
H_r_full = Xlist_to_Hlist(X_rall)  # (ntasks_rech, 4, 4)
tspace_dist = se3_error_pairwise_distance(H_r_full, 0.2)  # (ntasks_r, ntasks_r)
print(f"==>> tspace_dist.shape: \n{tspace_dist.shape}")
# -----------------------------------------------------------------


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
                group_matrix = find_altconfig_redudancy_fast(Qt1, Qt2)[3]
                cspace_group[idx, i0:i1, j0:j1] = group_matrix


# constants for GTSP
epslGH = 1.0
eta_collision = 0.1  # resolution of collision checking
cmax_d = 2 * np.pi  # max distance in cspace

cspace_eudist_v = cspace_eudist <= cmax_d
cspace_eudist_f = np.where(cspace_eudist_v, cspace_eudist, np.nan)
print(f"==>> cspace_eudist_f.shape: \n{cspace_eudist_f.shape}")


fd_sim = filter_cspace_candidate_similar_to_qinit(Qaik_r, qinit, thresh_mult=0.08)
fd_r = filter_cspace_candidate_radius_to_qinit(Qaik_r, qinit, radius=2 * np.pi)
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


I = 1
J = 5
qi_in_task_radius_I = qi_in_task_radius[task_ids_radius == I]
qi_in_task_radius_J = qi_in_task_radius[task_ids_radius == J]
print(f"==>> qi_in_task_radius_I: \n{qi_in_task_radius_I}")
print(f"==>> qi_in_task_radius_J: \n{qi_in_task_radius_J}")

CIJ = query_tasks_data_from_tspace_map(I, J, cspace_eudist, task_to_nn_unique)
print(f"==>> CIJ.shape: \n{CIJ.shape}")
print(f"==>> CIJ: \n{CIJ}")

CIJ_F = query_tasks_data_from_tspace_map(I, J, cspace_eudist_f, task_to_nn_unique)
print(f"==>> CIJ_F.shape: \n{CIJ_F.shape}")
print(f"==>> CIJ_F: \n{CIJ_F}")

CIJ_val_e = query_tasks_data_from_tspace_map(
    I, J, cspace_eudist_v, task_to_nn_unique
)
print(f"==>> CIJ_val_e.shape: \n{CIJ_val_e.shape}")
print(f"==>> CIJ_val_e: \n{CIJ_val_e}")

U, V = np.where(CIJ)
print(f"==>> U: \n{U}")
print(f"==>> V: \n{V}")

qi_to_keep_I = np.intersect1d(qi_in_task_radius_I, U.astype(int))
print(f"==>> qi_to_keep_I: \n{qi_to_keep_I}")
qi_to_keep_J = np.intersect1d(qi_in_task_radius_J, V.astype(int))
print(f"==>> qi_to_keep_J: \n{qi_to_keep_J}")

Qaik_rall_I = Qaik_rall[I]  # (n_ik*altcnf, dof)
print(f"==>> Qaik_rall_I: \n{Qaik_rall_I}")
Qaik_rall_J = Qaik_rall[J]  # (n_ik*altcnf, dof)
print(f"==>> Qaik_rall_J: \n{Qaik_rall_J}")
