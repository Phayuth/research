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
    H_to_X,
    xlist_to_Xlist,
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    se3_error_pairwise_distance,
)
from paper_sequential_planner.experiments.env_ur5e import RobotUR5eKin

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robkin = RobotUR5eKin()


def sample_reachable_wspace(ntasks):
    Hlist = []
    for _ in range(ntasks):
        q = np.random.uniform(-np.pi, np.pi, size=(6,))
        H = robkin.solve_fk(q)
        Hlist.append(H)
    X = Hlist_to_Xlist(Hlist)
    return X


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
# ntasts = 50
# X = sample_reachable_wspace(ntasts)
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

# 300 tasks OVER MEMORY LIMIT, SO ANNOYING
# cspace distance by IK pairing------------------------------------
# shape (ntasks_rech, ntasks_rech, n_ik * altcnf, n_ik * altcnf)
# accessing same id will give us dist to itself, which we dont need.
# always access different task id to get task-to-task distance, which we need
ntasks_rech, n_ik, dof = Qaik_rall.shape
_Qflat = Qaik_rall.reshape(ntasks_rech * n_ik, dof)
_cspace_eudist_flat = nan_euclidean_distances(_Qflat, _Qflat)
cspace_eudist = _cspace_eudist_flat.reshape(ntasks_rech, n_ik, ntasks_rech, n_ik)
cspace_eudist = cspace_eudist.transpose(0, 2, 1, 3)
print(f"==>> cspace_eudist.shape: \n{cspace_eudist.shape}")
# -----------------------------------------------------------------


# maybe not useful yet
# task-to-task distance by best IK pairing -> (ntasks_rech, ntasks_rech) -------
_cspace_dist_inf = np.where(np.isnan(cspace_eudist), np.inf, cspace_eudist)
cspace_task_min = _cspace_dist_inf.min(axis=(2, 3))
cspace_task_min[~np.isfinite(cspace_task_min)] = np.nan
print(f"==>> cspace_task_min.shape: \n{cspace_task_min.shape}")
# -----------------------------------------------------------------


# task-to-task distance by best IK pairing -> shape (ntasks_rech, ntasks_rech, 2,)
min_flat_idx = np.argmin(
    _cspace_dist_inf.reshape(
        _cspace_dist_inf.shape[0], _cspace_dist_inf.shape[1], -1
    ),
    axis=2,
)
min_idx_2d = np.unravel_index(min_flat_idx, _cspace_dist_inf.shape[2:])
cspace_task_min_idx = np.stack(min_idx_2d, axis=-1)
print(f"==>> cspace_task_min_idx.shape: \n{cspace_task_min_idx.shape}")
# -----------------------------------------------------------------


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


# task space neighbors by radius and knn, then union them ---------
def task_space_correlation():
    nnr = 0.5
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


nn_union, nn_dist, nn_count, nn_r, nn_k = task_space_correlation()
print(f"==>> nn_count: \n{nn_count}")
# -----------------------------------------------------------------


# contruct taskspace graph connection adjmat from nnunion ------------
# row (from), col (to), value (connected or not)
task_to_task_adj = np.zeros_like(tspace_dist, dtype=bool)
for i in range(tspace_dist.shape[0]):
    for j in nn_union[i]:
        task_to_task_adj[i, j] = True
        task_to_task_adj[j, i] = True  # undirected
print(f"==>> task_to_task_adj.shape: \n{task_to_task_adj.shape}")


# from this adjmat, we eliminate the task-to-task q pairs
# not need since i can just use the task-to-task adjmat
def task_to_task_cspace_adj():
    c_to_c_adj = np.zeros_like(cspace_eudist, dtype=bool)
    for i in range(task_to_task_adj.shape[0]):
        for j in range(i + 1, task_to_task_adj.shape[0]):
            if task_to_task_adj[i, j]:
                c_to_c_adj[i, j] = True
                c_to_c_adj[j, i] = True  # undirected
    return c_to_c_adj


# unique groups of IK ----------------------------------------------
def make_group_matrix():
    ik_num = 8
    altc_num = 32
    group_mat = np.zeros_like(cspace_eudist, dtype=int)
    for ti in range(task_to_task_adj.shape[0]):
        for tj in range(ti + 1, task_to_task_adj.shape[0]):
            if task_to_task_adj[ti, tj]:
                for ik_i in range(ik_num):
                    for ik_j in range(ik_num):
                        i0 = ik_i * altc_num
                        j0 = ik_j * altc_num
                        i1 = ik_i * altc_num + altc_num
                        j1 = ik_j * altc_num + altc_num
                        DIST = cspace_eudist[ti, tj, i0:i1, j0:j1]
                        if np.isnan(DIST).all():
                            group_mat[ti, tj, i0:i1, j0:j1] = -1
                        else:
                            group_pairs, groups_num, total_pairs, groups_matrix = (
                                find_altconfig_redudancy(
                                    Qaik_rall[ti, i0:i1],
                                    Qaik_rall[tj, j0:j1],
                                    DIST,
                                )
                            )
                            group_mat[ti, tj, i0:i1, j0:j1] = groups_matrix
    return group_mat


group_mat = make_group_matrix()
print(f"==>> group_mat.shape: \n{group_mat.shape}")
# -----------------------------------------------------------------


# bulk processing for all task pairs --------------------------------------
max_allow_cspace_dist = 2 * np.pi
cspace_eudist_filtermax = cspace_eudist <= max_allow_cspace_dist
t_to_t_adj_proc = np.zeros_like(task_to_task_adj, dtype=bool)  # track processed

# estimate the cost of real cspace
cspace_eudist_estimated = np.full_like(cspace_eudist, -1)
ik_num = 8  # 8 ik solutions
altc_num = 32  # 32 alt config per ik solution
ntasks_rech = task_to_task_adj.shape[0]
for ti in range(ntasks_rech):
    for tj in range(ti + 1, ntasks_rech):
        if task_to_task_adj[ti, tj]:  # check if this edge is considered
            if not t_to_t_adj_proc[ti, tj]:  # check if it processed yet
                # mark as processed
                t_to_t_adj_proc[ti, tj] = True
                t_to_t_adj_proc[tj, ti] = True  # undirected

                for ik_i in range(ik_num):
                    for ik_j in range(ik_num):
                        i0 = ik_i * altc_num
                        j0 = ik_j * altc_num
                        i1 = ik_i * altc_num + altc_num
                        j1 = ik_j * altc_num + altc_num

                        lowboud = cspace_eudist[ti, tj, i0:i1, j0:j1]

                        if np.isnan(lowboud).all():  # invalid pair, skip
                            cspace_eudist_estimated[ti, tj, i0:i1, j0:j1] = np.nan
                        else:
                            g = group_mat[ti, tj, i0:i1, j0:j1]
                            p = cspace_eudist_filtermax[ti, tj, i0:i1, j0:j1]
                            g_valid = g[p]
                            g_unique, first_valid_idx, count = np.unique(
                                g_valid, return_counts=True, return_index=True
                            )
                            valid_coords = np.column_stack(np.where(p))
                            g_unique_ij = valid_coords[first_valid_idx].astype(int)

                            # recovery of full-matrix indices from block-local idx
                            uv = g_unique_ij + np.array([i0, j0], dtype=int)

                            # recover config value from group id
                            Qs = Qaik_rall[ti, uv[:, 0]]  # start config
                            Qg = Qaik_rall[tj, uv[:, 1]]  # goal config

                            cost_group_est = np.full_like(
                                g_unique, np.inf, dtype=float
                            )
                            # for each pair of qs and qg, we estimate cost
                            for idx, (q_s, q_g) in enumerate(zip(Qs, Qg)):
                                cost = (
                                    np.linalg.norm(q_s - q_g)
                                    + 0.2 * np.random.uniform()
                                )
                                cost_group_est[idx] = cost

                            # assign the estimated cost to all pairs in the same group
                            g_cost_est = np.full_like(g, np.nan, dtype=float)
                            for idx, g_id in enumerate(g_unique):
                                g_cost_est[g == g_id] = cost_group_est[idx]
                            cspace_eudist_estimated[ti, tj, i0:i1, j0:j1] = (
                                g_cost_est
                            )
                            cspace_eudist_estimated[tj, ti, j0:j1, i0:i1] = (
                                g_cost_est.T
                            )  # undirected
print(f"==>> cspace_eudist_estimated.shape: \n{cspace_eudist_estimated.shape}")


# def write_to_GTSP_format(cspace_eudist_estimated):
#     ntasks_rech = cspace_eudist_estimated.shape[0]
#     ik_num = cspace_eudist_estimated.shape[2]

#     # with open("filename", "w") as f:
#     #     f.write(f"NAME: random_gtsp_fullmatrix\n")
#     #     f.write(f"TYPE: GTSP\n")
#     #     f.write(f"COMMENT: generated GTSP/AGTSP instance with full matrix\n")
#     #     f.write(f"DIMENSION: {num_points}\n")
#     #     f.write(f"GTSP_SETS: {num_clusters}\n")
#     #     f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
#     #     f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
#     #     f.write(f"EDGE_WEIGHT_SECTION\n")

#     #     # --- write all points ---
#     #     for i in range(num_points):
#     #         row = " ".join(f"{matrix[i, j]}" for j in range(num_points))
#     #         f.write(f"{row}\n")

#     #     # --- write GTSP sets ---
#     #     f.write("GTSP_SET_SECTION\n")
#     #     for key in clusters.keys():
#     #         c = clusters[key]
#     #         k = key
#     #         nodes_str = " ".join(str(n + 1) for n in c)
#     #         f.write(f"{k + 1} {nodes_str} -1\n")

#     #     f.write("EOF\n")


def visualize():
    Hhome = H_r_full[0]

    t1 = 0
    Hnnt1 = nn_union[t1]
    Hnn = [H_r_full[i] for i in Hnnt1]

    ax = make_3d_axis(1)
    plot_transform(ax, np.eye(4), s=1, name="world")
    plot_transform(ax, Hhome, s=0.1, c="r", name="home")
    for i in range(H_r_full.shape[0]):
        plot_transform(ax, H_r_full[i], s=0.05, name=f"{i}")
    for h in Hnn:
        ax.plot(
            [Hhome[0, 3], h[0, 3]],
            [Hhome[1, 3], h[1, 3]],
            [Hhome[2, 3], h[2, 3]],
            "k--",
            linewidth=0.5,
        )
        plot_transform(ax, h, s=0.1, c="g", name=f"n")
    plt.show()


visualize()
