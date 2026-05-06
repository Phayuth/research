import os
import numpy as np
from paper_sequential_planner.scripts.geometric_torus import (
    find_alt_config2,
    find_altconfig_redudancy_wrong,
    find_altconfig_redudancy,
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
    task_space_correlation_mapping,
)
from paper_sequential_planner.experiments.env_ur5e import RobotUR5eKin

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robkin = RobotUR5eKin()
scene = None  # for collision checking, not implemented yet
planner = None  # placeholder for planner instance, not implemented yet


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
    alt_num = 32
    unique_sols = 8
    num_sols = unique_sols * alt_num
    dof = 6
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
nn_union, nn_dist, nn_count, nn_r, nn_k = (
    tspace_coorrelation["nn_union"],
    tspace_coorrelation["nn_dist"],
    tspace_coorrelation["nn_count"],
    tspace_coorrelation["nn_r"],
    tspace_coorrelation["nn_k"],
)


tspace_mapping = task_space_correlation_mapping(tspace_coorrelation)
task_to_nn_mapping, task_to_nn_unique, task_to_nn_unique_len = (
    tspace_mapping["task_to_nn_mapping"],
    tspace_mapping["task_to_nn_unique"],
    tspace_mapping["task_to_nn_unique_len"],
)
print(f"==>> task_to_nn_mapping: \n{task_to_nn_mapping}")
print(f"==>> task_to_nn_unique (i<j): \n{task_to_nn_unique}")
print(f"==>> number of unique undirected edges: \n{task_to_nn_unique_len}")


ntasks_rech, n_ik, dof = Qaik_rall.shape
print(f"==>> ntasks_rech: \n{ntasks_rech}")
print(f"==>> n_ik: \n{n_ik}")
print(f"==>> dof: \n{dof}")


cspace_eudist = np.empty((task_to_nn_unique_len, n_ik, n_ik))
for idx, (i, j) in enumerate(task_to_nn_unique):
    Qt1 = Qaik_rall[i]  # (n_ik*altcnf, dof)
    Qt2 = Qaik_rall[j]  # (n_ik*altcnf, dof)
    CspaceDistT1T2 = nan_euclidean_distances(Qt1, Qt2)
    cspace_eudist[idx] = CspaceDistT1T2
print(f"==>> cspace_eudist.shape: \n{cspace_eudist.shape}")


def query_cspace_distance(i, j, cspace_eudist):
    """
    query (I, J) and (J, I) give the same value
    but the shape is transposed to swap T1 and T2
    """
    if j < i:
        a, b = (i, j) if i < j else (j, i)
        idx = task_to_nn_unique.index((a, b))
        return cspace_eudist[idx].T  # transpose to swap T1 and T2
    else:
        a, b = (i, j) if i < j else (j, i)
        idx = task_to_nn_unique.index((a, b))
        return cspace_eudist[idx]


I = 0
J = 10
CspaceDistIJ = query_cspace_distance(I, J, cspace_eudist)
CspaceDistJI = query_cspace_distance(J, I, cspace_eudist)
print(f"==>> CspaceDist({I}, {J}).shape: \n{CspaceDistIJ.shape}")
print(f"==>> CspaceDist({J}, {I}).shape: \n{CspaceDistJI.shape}")


# constants for GTSP
epslGH = 1.0
eta_collision = 0.1  # resolution of collision checking
max_allow_cspace_dist = 2 * np.pi  # max distance in cspace

cspace_eudist_filtermax = cspace_eudist <= max_allow_cspace_dist
cspace_eudist_filtered = np.where(cspace_eudist_filtermax, cspace_eudist, np.nan)
# print(f"==>> cspace_eudist_filtered: \n{cspace_eudist_filtered}")

# # how many valid edges after filtering by cspace distance?
# for idx in range(task_to_nn_unique_len):
#     num_valid_edges = np.sum(~np.isnan(cspace_eudist_filtered[idx]))
#     print(f"==>> number of valid edges for edge {task_to_nn_unique[idx]}: \n{num_valid_edges}")

proc_task = [0] * task_to_nn_unique_len
for idx, (i, j) in enumerate(task_to_nn_unique):
    d = query_cspace_distance(i, j, cspace_eudist_filtered)
    u, v = np.where(~np.isnan(d))
    print(f"==>> u: \n{u}")
    print(f"==>> v: \n{v}")
