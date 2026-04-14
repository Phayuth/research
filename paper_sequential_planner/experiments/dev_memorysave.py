import os
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform
from paper_sequential_planner.scripts.rtsp_solver import RTSP
from paper_sequential_planner.scripts.geometric_torus import (
    find_alt_config2,
    find_altconfig_redudancy,
)
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_poses import (
    poses_a,
    poses_b,
    poses_c,
    poses_d,
    poses_epGH,
    H_to_X,
    xlist_to_Xlist,
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    se3_error_pairwise_distance,
)
from paper_sequential_planner.experiments.env_ur5e import RobotUR5eKin
import pprint

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robkin = RobotUR5eKin()


def radius_neighbors(D, radius):
    neighbors = []
    for i in range(D.shape[0]):
        idx = np.where(D[i] < radius)[0]
        idx = idx[idx != i]  # remove self
        neighbors.append(idx.tolist())
    return neighbors


def knn_from_distance(D, k=5):
    # ignore self-distance by setting diagonal large
    D = D.copy()
    np.fill_diagonal(D, np.inf)

    idx = np.argpartition(D, k, axis=1)[:, :k]  # (N, k)

    # optional: sort neighbors by distance
    row_idx = np.arange(D.shape[0])[:, None]
    sorted_order = np.argsort(D[row_idx, idx], axis=1)
    idx = idx[row_idx, sorted_order]

    return idx.tolist()  # indices of k nearest per row


def task_space_correlation():
    nnr = 0.15
    nnk = 5
    nn_r = radius_neighbors(tspace_dist, radius=nnr)
    nn_k = knn_from_distance(tspace_dist, k=nnk)
    nn_union = []
    for i in range(tspace_dist.shape[0]):
        union_set = set(nn_r[i]) | set(nn_k[i])
        nn_union.append(sorted(union_set))
    nn_dist = []
    for i in range(len(nn_union)):
        dists = [tspace_dist[i, j].item() for j in nn_union[i]]
        nn_dist.append(dists)

    nn_count = [len(n) for n in nn_union]
    return nn_union, nn_dist, nn_count, nn_r, nn_k


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
# H = poses_epGH()
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


nn_union, nn_dist, nn_count, nn_r, nn_k = task_space_correlation()
print(f"==>> nn_union: \n{nn_union}")
print(f"==>> nn_count: \n{nn_count}")

t1 = 1
t1nn = nn_union[t1]
t2 = t1nn[0]
print(f"==>> t1: \n{t1}")
print(f"==>> t1nn: \n{t1nn}")
print(f"==>> t2: \n{t2}")

# Qt1 = Qaik_rall[t1]  # (n_ik*altcnf, dof)
# Qt2 = Qaik_rall[t2]  # (n_ik*altcnf, dof)
# print(f"==>> Qt1.shape: \n{Qt1.shape}")
# print(f"==>> Qt1: \n{Qt1}")
# print(f"==>> Qt2.shape: \n{Qt2.shape}")
# print(f"==>> Qt2: \n{Qt2}")

# CspaceDistT1T2 = nan_euclidean_distances(Qt1, Qt2)
# print(f"==>> CspaceDistT1T2: \n{CspaceDistT1T2}")

# CspaceDistT2T1 = nan_euclidean_distances(Qt2, Qt1)
# print(f"==>> CspaceDistT2T1: \n{CspaceDistT2T1}")

# C = np.isclose(CspaceDistT1T2, CspaceDistT2T1)  # should be symmetric
# print(f"==>> C: \n{C}")


# k-NN: directed graph (not mutual)
# r-NN: undirected graph (mutual, under symmetric metric)
# so we most likely dont find mutual k-NN
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
print(f"==>> edges_unique (i<j): \n{edges}")


# assign a value to each undirected edge (here: task-space distance)
edge_value = {(a, b): np.random.randint(10) for (a, b) in edges}
print(f"==>> edge_value: \n{edge_value}")


def query_edge_value(i, j, value_map):
    if j < i:
        a, b = (i, j) if i < j else (j, i)
        return value_map.get((a, b), None)*1000
    else:
        a, b = (i, j) if i < j else (j, i)
        return value_map.get((a, b), None)


# example: query (I, J) and (J, I) give the same value
I, J = 0, 10
vij = query_edge_value(I, J, edge_value)
vji = query_edge_value(J, I, edge_value)
print(f"==>> query ({I}, {J}) = {vij}")
print(f"==>> query ({J}, {I}) = {vji}")
