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
# -----------------------------------------------------------------

# to visit task ---------------------------------------------------
H = poses_d()
X = Hlist_to_Xlist(H)
Qaik = wspace_ik_extended(robkin, X)  # (ntasks, 7)
Qaik_valid = wspace_ik_validity_extended(Qaik, None)  # (ntasks, 7, 1)
# -----------------------------------------------------------------

# filterout reachable/unreachable tasks ---------------------------
X_isunr = np.all(Qaik_valid != 1, axis=1).flatten()  # (ntasks, ) True/False
X_r = X[~X_isunr]  # (ntasks_rech, 2)
Qaik_valid_r = Qaik_valid[~X_isunr]  # (ntasks_rech, n_ik * altcnf, 1)
Qaik_r = Qaik[~X_isunr]  # (ntasks_rech, n_ik * altcnf, dof)
Qaik_r = np.where(Qaik_valid_r == 1, Qaik_r, np.nan)  # set value to nan if invalid
# -----------------------------------------------------------------

# concate Xinit, qinit ------------------------------------------------
X_rall = np.vstack((Xinit, X_r))  # (ntasks+1, 2) & init
qinit_all = np.full((1, Qaik_r.shape[1], Qaik_r.shape[2]), np.nan)
qinit_all[0, 0] = qinit  # set first IK to qinit and rest to nan, real val no alt
Qaik_rall = np.vstack((qinit_all, Qaik_r))  # (ntasks+1, n_ik*altcnf, dof) & init
print(f"==>> X_rall.shape: \n{X_rall.shape}")
print(f"==>> Qaik_rall.shape: \n{Qaik_rall.shape}")
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
print(f"==>> nn_union: \n{nn_union}")
print(f"==>> nn_dist: \n{nn_dist}")
print(f"==>> nn_count: \n{nn_count}")
print(f"==>> nn_r: \n{nn_r}")
print(f"==>> nn_k: \n{nn_k}")


nnn = {}
for i in range(len(nn_union)):
    for j in nn_union[i]:
        nnn[i] = nnn.get(i, []) + [j]
print(f"==>> nnn: \n{nnn}")

# unique undirected edges in canonical order: (i, j) with i < j
edges_set = set()
for i in range(len(nn_union)):
    for j in nn_union[i]:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        edges_set.add((a, b))

edges = sorted(edges_set)
num_uniq_tedges = len(edges)
print(f"==>> edges_unique (i<j): \n{edges}")
print(f"==>> number of unique undirected edges: \n{num_uniq_tedges}")

# assign a value to each undirected edge (here: task-space distance)
edge_value = {(a, b): np.random.randint(10) for (a, b) in edges}
print(f"==>> edge_value: \n{edge_value}")


def query_edge_value(i, j, data):
    if j < i:
        a, b = (i, j) if i < j else (j, i)
        return data.get((a, b), None) * 1000
    else:
        a, b = (i, j) if i < j else (j, i)
        return data.get((a, b), None)


# example: query (I, J) and (J, I) give the same value
I, J = 0, 10
vij = query_edge_value(I, J, edge_value)
vji = query_edge_value(J, I, edge_value)
print(f"==>> query ({I}, {J}) = {vij}")
print(f"==>> query ({J}, {I}) = {vji}")


ntasks_rech, n_ik, dof = Qaik_rall.shape
print(f"==>> ntasks_rech: \n{ntasks_rech}")
print(f"==>> n_ik: \n{n_ik}")
print(f"==>> dof: \n{dof}")


cspace_eudist = np.empty((num_uniq_tedges, n_ik, n_ik))
for idx, (i, j) in enumerate(edges):
    Qt1 = Qaik_rall[i]  # (n_ik*altcnf, dof)
    Qt2 = Qaik_rall[j]  # (n_ik*altcnf, dof)
    CspaceDistT1T2 = nan_euclidean_distances(Qt1, Qt2)
    cspace_eudist[idx] = CspaceDistT1T2
print(f"==>> cspace_eudist.shape: \n{cspace_eudist.shape}")


def query_cspace_distance(i, j, cspace_eudist):
    if j < i:
        a, b = (i, j) if i < j else (j, i)
        idx = edges.index((a, b))
        return cspace_eudist[idx].T  # transpose to swap T1 and T2
    else:
        a, b = (i, j) if i < j else (j, i)
        idx = edges.index((a, b))
        return cspace_eudist[idx]


CspaceDistIJ = query_cspace_distance(I, J, cspace_eudist)
CspaceDistJI = query_cspace_distance(J, I, cspace_eudist)
print(f"==>> CspaceDist({I}, {J}).shape: \n{CspaceDistIJ.shape}")
print(f"==>> CspaceDist({J}, {I}).shape: \n{CspaceDistJI.shape}")
