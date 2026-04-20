import os
import numpy as np
import matplotlib.pyplot as plt
from paper_sequential_planner.experiments.env_planarrr import (
    PlanarRR,
    RobotScene,
    OMPLPlanner,
    sample_reachable_wspace,
    wspace_ik_extended,
    wspace_ik_validity_extended,
)
from paper_sequential_planner.scripts.rtsp_solver import RTSP
from paper_sequential_planner.scripts.geometric_torus import (
    find_alt_config2,
    find_altconfig_redudancy,
)
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances
from paper_sequential_planner.scripts.geometric_poses import (
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    xlist_to_Xlist,
    se3_error,
    se3_error_pairwise_distance,
)
from paper_sequential_planner.scripts.geometric_ellipse import *

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robot = PlanarRR()
scene = RobotScene(robot, None)
planner = OMPLPlanner(scene.collision_checker)


# initial config and task  -----------------------------------------
limit2 = np.array(
    [
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
    ]
)
qinit = np.array([1, -1])
Xinit = np.array([robot.forward_kinematic(qinit)[-1]])
# -----------------------------------------------------------------

# to visit task ---------------------------------------------------
ntasks = 30
X = sample_reachable_wspace(ntasks)  # (ntasks, 2)
Qaik = wspace_ik_extended(robot, X)  # (ntasks, n_ik * altcnf, dof)
Qaik_valid = wspace_ik_validity_extended(Qaik, scene)  # (ntasks, n_ik * altcnf, 1)
# -----------------------------------------------------------------

# Seed workspace task and their IK for cost estimation ------------
NSEED = 300
XSEED = sample_reachable_wspace(NSEED)  # (NSEED, 2)
QSEEDAIK = wspace_ik_extended(robot, XSEED)  # (NSEED, n_ik * altcnf, dof)
QSEEDAIK_Valid = wspace_ik_validity_extended(QSEEDAIK, scene)
# -----------------------------------
XSEED_unr = np.all(QSEEDAIK_Valid != 1, axis=1).flatten()  # (NSEED, ) True/False
XSEED_r = XSEED[~XSEED_unr]  # (NSEED_rech, 2)
QSEEDAIK_valid_r = QSEEDAIK_Valid[~XSEED_unr]  # (NSEED_rech, n_ik * altcnf, dof)
QSEEDAIK_r = QSEEDAIK[~XSEED_unr]
QSEEDAIK_r = np.where(QSEEDAIK_valid_r == 1, QSEEDAIK_r, np.nan)
print(f"==>> QSEEDAIK_r.shape: \n{QSEEDAIK_r.shape}")
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
X_r_full = xlist_to_Xlist(X_rall)  # (ntasks_rech, 6)
H_r_full = Xlist_to_Hlist(X_r_full)  # (ntasks_rech, 4, 4)
tspace_dist = se3_error_pairwise_distance(H_r_full, 0.2)  # (ntasks_r, ntasks_r)
print(f"==>> tspace_dist.shape: \n{tspace_dist.shape}")
# -----------------------------------------------------------------


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


def task_to_task_configuration_interp(Qaik_rall, nintp=10):
    """
    Interpolate between all pairs of task-reachable configurations,
    and set to nan if either of the pair is invalid.
    final shape (ntasks_rech, ntasks_rech, n_ik, n_ik, nintp, dof)
    Ex:
    tt_cspace_interp = task_to_task_configuration_interp(Qaik_rall, nintp=10)
    t0t1q0q0 = tt_cspace_interp[0, 1, 0, 0]
    give us interp bet/ first IK of task0 to first IK of task1.
    task id should not be the same
    """
    ntasks_rech, n_ik, dof = Qaik_rall.shape
    Q1 = Qaik_rall[:, None, :, None, None, :]  # (ntasks_rech,1,n_ik,1,1,dof)
    Q2 = Qaik_rall[None, :, None, :, None, :]  # (1,ntasks_rech,1,n_ik,1,dof)
    tau = np.linspace(0.0, 1.0, nintp, dtype=Qaik_rall.dtype)
    tau = tau[None, None, None, None, :, None]  # (1,1,1,1,nintp,1)
    # (ntasks_rech,ntasks_rech,n_ik,n_ik,nintp,dof)
    interp = (1.0 - tau) * Q1 + tau * Q2
    invalid_cfg = np.isnan(Qaik_rall).all(axis=-1)  # (ntasks_rech,n_ik)
    # (ntasks_rech,ntasks_rech,n_ik,n_ik)
    invalid_pair = invalid_cfg[:, None, :, None] | invalid_cfg[None, :, None, :]
    interp = np.where(invalid_pair[..., None, None], np.nan, interp)
    return interp


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
    ik_num = 2  # elbow up and down
    altc_num = 4  # 4 alt config per ik solution
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


t1 = 1
t1nn = nn_union[t1]
t2 = t1nn[0]
print(f"==>> t1: \n{t1}")
print(f"==>> t1nn: \n{t1nn}")
print(f"==>> t2: \n{t2}")

# thinking of doing isometry distance
epslGH = 1.0
eta_collision = 0.1  # resolution of collision checking
# t1_nnnn = nn_dist[t1]
# print(f"==>> t1_nnnn: \n{t1_nnnn}")
# t1nntn = t1_nnnn[0]
# print(f"==>> t1nntn: \n{t1nntn}")
# ts_cs_diff = t1cpc - t1nntn
# print(f"==>> ts_cs_diff: \n{ts_cs_diff}")


# bulk processing for all task pairs --------------------------------------
max_allow_cspace_dist = 2 * np.pi
cspace_eudist_filtermax = cspace_eudist <= max_allow_cspace_dist
cspace_eudist_filtered = np.where(cspace_eudist_filtermax, cspace_eudist, np.nan)
t_to_t_adj_proc = np.zeros_like(task_to_task_adj, dtype=bool)  # track processed

# estimate the cost of real cspace
cspace_eudist_estimated = np.full_like(cspace_eudist, -1)
ik_num = 2  # elbow up and down
altc_num = 4  # 4 alt config per ik solution
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
                                    + 0.0 * np.random.uniform()
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


def weighted_euclidean_distance():
    """weighted euclidean distance to initial config,
    as a way to select subset of configurations for each task
    """
    ntasks_rech, n_ik, dof = Qaik_r.shape
    Qaik_r_flat = Qaik_r.reshape(ntasks_rech * n_ik, dof)
    cdd = nan_euclidean_distances(Qaik_r_flat, qinit.reshape(1, -1))
    cdd_recp = 1.0 / (cdd + 0.001)  # add small value to avoid div by zero
    cdd_sum = np.nansum(cdd_recp)  # use nansum to ignore nan values
    cdd_ratio = cdd_recp / cdd_sum
    cdd_ratio_min = np.nanmin(cdd_ratio)
    cdd_ratio_max = np.nanmax(cdd_ratio)
    threshold = cdd_ratio_min + 0.5 * (
        cdd_ratio_max - cdd_ratio_min
    )  # select configs above the midpoint
    print(f"min: {cdd_ratio_min}, max: {cdd_ratio_max}, threshold: {threshold}")
    get_Qind_inthreshold = cdd_ratio >= threshold
    Qin_threshold = Qaik_r_flat[get_Qind_inthreshold.flatten()]
    print(f"==>> Qin_threshold: \n{Qin_threshold}")
    return Qin_threshold


Qin_threshold = weighted_euclidean_distance()


def GTSP_WRITE(
    cspace_eudist_estimated,
    filename="problem_dev.gtsp",
    cost_scale=1000,
    no_edge_weight=None,
):
    """
    Write GTSP instance for GLKH from 4D cost tensor.

    Input shape must be (ntask, ntask, node, node).
    np.nan values are treated as disconnected edges and replaced by a large integer.
    """
    if cspace_eudist_estimated.ndim != 4:
        raise ValueError("Expected shape (ntask, ntask, node, node)")

    ntask, ntask_2, node_i, node_j = cspace_eudist_estimated.shape
    if ntask != ntask_2 or node_i != node_j:
        raise ValueError(
            "Expected shape (ntask, ntask, node, node) with square blocks"
        )

    nodes_per_task = node_i
    dimension = ntask * nodes_per_task

    # Convert 4D tensor to a full matrix ordered by task-major node index.
    # global_id = task_id * nodes_per_task + node_id
    matrix = cspace_eudist_estimated.transpose(0, 2, 1, 3).reshape(
        dimension, dimension
    )

    # Treat negative placeholders (e.g., -1) as disconnected edges.
    finite_mask = np.isfinite(matrix) & (matrix >= 0.0)
    scaled = np.full((dimension, dimension), np.nan, dtype=float)
    scaled[finite_mask] = np.rint(matrix[finite_mask] * cost_scale)

    if no_edge_weight is None:
        if np.any(finite_mask):
            max_finite = int(np.nanmax(scaled[finite_mask]))
            no_edge_weight = max(10**7, max_finite * 100 + 1)
        else:
            no_edge_weight = 10**7

    weight_mat = np.where(np.isfinite(scaled), scaled, no_edge_weight).astype(
        np.int64
    )
    np.fill_diagonal(weight_mat, 0)

    with open(filename, "w") as f:
        f.write("NAME: rtsp_problem\n")
        f.write("TYPE: GTSP\n")
        f.write("COMMENT: rtsp problem with torus space\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write(f"GTSP_SETS: {ntask}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        for i in range(dimension):
            row = " ".join(str(weight_mat[i, j]) for j in range(dimension))
            f.write(f"{row}\n")

        f.write("GTSP_SET_SECTION\n")
        for task_id in range(ntask):
            start = task_id * nodes_per_task + 1  # GLKH is 1-based
            end = start + nodes_per_task
            nodes = " ".join(str(n) for n in range(start, end))
            f.write(f"{task_id + 1} {nodes} -1\n")

        f.write("EOF\n")

    print(f"==>> wrote GTSP file: {filename}")
    print(
        f"==>> dimension: {dimension}, sets: {ntask}, no_edge_weight: {no_edge_weight}"
    )
    return filename


def GTSP_WRITE_INDEX_MAPPING():
    pass


def GTSP_LOAD():
    """
    Load GLKH tour file and map flattened node IDs back to (task_id, node_id).

    Returns a dict with:
      - tour_global_0based: list[int]
      - tour_task_node: list[(task_local, node_local)]
      - tour_task_original: list[int]  # -1 means init task
      - tour_q: np.ndarray, shape (len(tour), dof)
    """
    filename = os.path.join(rsrc, "GLKH-1.1", "PROBLEMS", "problem_dev.tour")

    node_ids_1based = []
    reading_tour = False
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "TOUR_SECTION":
                reading_tour = True
                continue
            if not reading_tour:
                continue
            if line in {"-1", "EOF"}:
                break
            try:
                node_ids_1based.append(int(line))
            except ValueError:
                # Ignore non-integer lines if any extra metadata appears.
                continue

    if len(node_ids_1based) == 0:
        raise ValueError(f"No tour node IDs found in {filename}")

    tour_global_0based = [n - 1 for n in node_ids_1based]
    # Close the loop if solver output did not include the return-to-start node.
    if tour_global_0based[0] != tour_global_0based[-1]:
        tour_global_0based.append(tour_global_0based[0])

    ntask_local, nodes_per_task, _ = Qaik_rall.shape
    dimension = ntask_local * nodes_per_task

    for g in tour_global_0based:
        if g < 0 or g >= dimension:
            raise ValueError(
                f"Tour node id {g} out of range [0, {dimension - 1}] for current Qaik_rall"
            )

    tour_task_node = []
    tour_q = []
    for g in tour_global_0based:
        task_local = g // nodes_per_task
        node_local = g % nodes_per_task
        tour_task_node.append((task_local, node_local))
        tour_q.append(Qaik_rall[task_local, node_local])
    tour_q = np.asarray(tour_q)

    # Map local task index (with init at 0) back to original task index in X.
    # Convention: -1 corresponds to init task.
    task_local_to_original = np.full(ntask_local, -1, dtype=int)
    if ntask_local > 1:
        task_local_to_original[1:] = np.where(~X_isunr)[0]
    tour_task_original = [task_local_to_original[t] for t, _ in tour_task_node]

    print(f"==>> loaded tour file: {filename}")
    print(f"==>> number of nodes in closed tour: {len(tour_global_0based)}")

    return {
        "tour_global_0based": tour_global_0based,
        "tour_task_node": tour_task_node,
        "tour_task_original": tour_task_original,
        "tour_q": tour_q,
    }


# GTSP_WRITE(
#     cspace_eudist_estimated,
#     filename=os.path.join(rsrc, "GLKH-1.1", "PROBLEMS", "problem_dev.gtsp"),
# )

# GTSP_WRITE_INDEX_MAPPING()

tour_data = None
tour_data = GTSP_LOAD()
qtour = tour_data["tour_q"]
print(f"==>> qtour: \n{qtour}")

Xtour = []
for q in qtour:
    x = robot.forward_kinematic(q)[-1]
    Xtour.append(x)
Xtour = np.array(Xtour)

# raise


def visualize():
    fig, ax = plt.subplots(1, 2)

    Xt1 = X_rall[t1]
    Xt2 = X_rall[t2]
    # obstacles
    for shp in scene.obstacles:
        x, y = shp.exterior.xy
        ax[0].fill(x, y, alpha=0.5, fc="red", ec="black")

    cirt1 = plt.Circle(
        Xt1, 0.5, color="r", fill=False, linestyle="--", label="t1 radius"
    )
    ax[0].add_artist(cirt1)
    cirt2 = plt.Circle(
        Xt2, 0.5, color="b", fill=False, linestyle="--", label="t2 radius"
    )
    ax[0].add_artist(cirt2)

    nnkt1 = nn_k[t1]
    nnrt1 = nn_r[t1]
    for nnt1 in nnkt1:
        ax[0].plot(
            [Xt1[0], X_rall[nnt1][0]],
            [Xt1[1], X_rall[nnt1][1]],
            "b--",
            label="t1 knn" if nnt1 == nnkt1[0] else "",
        )
    for nnt1 in nnrt1:
        ax[0].plot(
            [Xt1[0], X_rall[nnt1][0]],
            [Xt1[1], X_rall[nnt1][1]],
            "r--",
            label="t1 radius" if nnt1 == nnrt1[0] else "",
        )

    # ax0: Workspace
    links = np.array(robot.forward_kinematic(qinit))
    ax[0].text(Xinit[0, 0], Xinit[0, 1], "home task", fontsize=8, ha="right")
    ax[0].plot(links[:, 0], links[:, 1], "k-o", linewidth=2, label="Robot at q0")
    ax[0].plot(
        X[:, 0],
        X[:, 1],
        "o",
        color="lightgray",
        label="User Input Tasks",
    )
    ax[0].plot(
        X_rall[:, 0],
        X_rall[:, 1],
        "gx",
        label="Task-Reachable",
    )
    ax[0].plot(
        XSEED[:, 0],
        XSEED[:, 1],
        "k+",
        alpha=0.5,
        markersize=3,
        label="Seed Tasks",
    )
    for i, x in enumerate(X_rall):
        ax[0].text(x[0], x[1], f"({i})", fontsize=8, ha="right")
    if qtour is not None:
        ax[0].plot(
            Xtour[:, 0],
            Xtour[:, 1],
            "m-",
            linewidth=2,
            label="GTSP Tour in Workspace",
        )
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-4, 4)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
    )

    # process cspace
    # t2 is a global task id, but cspace_eudist[t1, t1nn] is indexed by local !!!
    # neighbor position. Map global id -> local neighbor index first.
    t2_local = t1nn.index(t2)
    print(f"==>> t2_local: \n{t2_local}")

    # all available edge
    cost_t1_to_t1nn = cspace_eudist[t1, t1nn]
    print(f"==>> cost_t1_to_t1nn: \n{cost_t1_to_t1nn[t2_local]}")
    notnan_index = np.where(~np.isnan(cost_t1_to_t1nn[t2_local]))
    idex = np.column_stack(notnan_index)
    Qs = Qaik_rall[t1, idex[:, 0]]  # start config
    Qg = Qaik_rall[t2, idex[:, 1]]  # goal config
    cost_all = cost_t1_to_t1nn[t2_local][idex[:, 0], idex[:, 1]]

    # edge to be considered
    cost_t1_to_t1nn_consd = cspace_eudist_filtered[t1, t1nn]
    print(f"==>> cost_t1_to_t1nn_consd: \n{cost_t1_to_t1nn_consd[t2_local]}")
    notnan_index_consd = np.where(~np.isnan(cost_t1_to_t1nn_consd[t2_local]))
    idex_consd = np.column_stack(notnan_index_consd)
    Qscond = Qaik_rall[t1, idex_consd[:, 0]]  # start config
    Qgconsd = Qaik_rall[t2, idex_consd[:, 1]]  # goal config
    cost_considered = cost_t1_to_t1nn_consd[t2_local][
        idex_consd[:, 0], idex_consd[:, 1]
    ]

    qscond = Qscond[2].reshape(-1, 1)
    qgcond = Qgconsd[2].reshape(-1, 1)
    cmin = np.linalg.norm(qscond - qgcond)
    cdiff = max_allow_cspace_dist - cmin
    print(f"==>> cdiff: \n{cdiff}")
    mulp = 1.3
    cMAx = cmin * mulp
    el = get_2d_ellipse_informed_mplpatch(qscond, qgcond, cMAx)
    ax[1].add_patch(el)
    xCenter, rotationAxisC, L, cMin = informed_sampling_ellipse(
        qscond, qgcond, cMAx
    )
    state = isPointinEllipseBulk2(
        xCenter, rotationAxisC, L, QSEEDAIK_r.reshape(-1, 2)
    )
    print(f"==>> state.shape: \n{state.shape}")

    QSEEDAIK_r_2d = QSEEDAIK_r.reshape(-1, 2)
    QINELLIPSE = QSEEDAIK_r_2d[state]

    Qaltconfighome = find_alt_config2(qinit, limit2, filterOriginalq=True)

    # ax1: C-space
    cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles_extended.npy"))
    ax[1].plot(
        [qinit[0]],
        [qinit[1]],
        "k-o",
        linewidth=2,
    )
    ax[1].text(qinit[0], qinit[1], "home config", fontsize=8, ha="right")
    ax[1].plot(
        Qaltconfighome[:, 0],
        Qaltconfighome[:, 1],
        "r^",
        markersize=10,
        label="Alt Home Config But not chosen (no teleport)",
    )

    ax[1].plot(
        cspace_obs[:, 0],
        cspace_obs[:, 1],
        "ro",
        markersize=1,
        label="C-space Obstacles",
        alpha=0.1,
    )
    ax[1].scatter(
        Qaik[:, :, 0].ravel(),
        Qaik[:, :, 1].ravel(),
        marker="o",
        color="lightgray",
        label="All IK Solutions",
    )
    ax[1].plot(
        Qaik_rall[:, :, 0].ravel(),
        Qaik_rall[:, :, 1].ravel(),
        "gx",
        markersize=5,
        label="Reachable IK Solutions",
    )
    # ax[1].plot(
    #     QSEEDAIK[:, :, 0].ravel(),
    #     QSEEDAIK[:, :, 1].ravel(),
    #     "k+",
    #     alpha=0.5,
    #     markersize=3,
    #     label="Seed IK Solutions",
    # )
    ax[1].plot(
        QINELLIPSE[:, 0],
        QINELLIPSE[:, 1],
        "bx",
        markersize=5,
        label="Seed IK in Ellipse",
    )
    if qtour is not None:
        ax[1].plot(
            qtour[:, 0],
            qtour[:, 1],
            "m-",
            linewidth=2,
            label="GTSP Tour in C-space",
        )

    # -------------hover interactivity for edge info display----------------
    def _fmt_vec(q):
        return np.array2string(q, precision=3, separator=", ")

    hover_lines = []
    hover_meta = {}
    for idx, (qs, qg) in enumerate(zip(Qs, Qg)):
        (line,) = ax[1].plot(
            [qs[0], qg[0]],
            [qs[1], qg[1]],
            "g--",
            alpha=0.5,
        )
        line.set_pickradius(2)
        hover_lines.append(line)
        hover_meta[line] = (
            "type: all available edge\n"
            f"cost distance: {cost_all[idx]:.4f}\n"
            f"start config: {_fmt_vec(qs)}\n"
            f"goal config: {_fmt_vec(qg)}"
        )

    for idx, (qs, qg) in enumerate(zip(Qscond, Qgconsd)):
        (line,) = ax[1].plot(
            [qs[0], qg[0]],
            [qs[1], qg[1]],
            "b--",
            alpha=0.5,
        )
        line.set_pickradius(10)
        hover_lines.append(line)
        hover_meta[line] = (
            "type: considered edge\n"
            f"cost distance: {cost_considered[idx]:.4f}\n"
            f"start config: {_fmt_vec(qs)}\n"
            f"goal config: {_fmt_vec(qg)}"
        )

    hover_annot = ax[1].annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    hover_annot.set_visible(False)
    active_line = {"line": None}

    def _set_hover_line(line):
        if active_line["line"] is line:
            return
        if active_line["line"] is not None:
            active_line["line"].set_linewidth(1.5)
            active_line["line"].set_alpha(0.5)
        if line is not None:
            line.set_linewidth(3.0)
            line.set_alpha(1.0)
        active_line["line"] = line

    def _on_move(event):
        if event.inaxes != ax[1]:
            if hover_annot.get_visible() or active_line["line"] is not None:
                _set_hover_line(None)
                hover_annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        for line in hover_lines:
            contains, _ = line.contains(event)
            if contains:
                _set_hover_line(line)
                hover_annot.xy = (event.xdata, event.ydata)
                hover_annot.set_text(hover_meta[line])
                hover_annot.set_visible(True)
                fig.canvas.draw_idle()
                return

        if hover_annot.get_visible() or active_line["line"] is not None:
            _set_hover_line(None)
            hover_annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    # -------------hover interactivity for edge info display----------------

    ax[1].set_aspect("equal")
    ax[1].set_xlim(-2 * np.pi, 2 * np.pi)
    ax[1].set_ylim(-2 * np.pi, 2 * np.pi)
    ax[1].set_xlabel("q1")
    ax[1].set_ylabel("q2")
    ax[1].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
    )
    plt.show()


import matplotlib.patches as patches


def visualize_torus():
    fig, ax = plt.subplots()

    Qt1 = Qaik_rall[t1]
    print(f"==>> Qt1: \n{Qt1}")
    Qt2 = Qaik_rall[t2]
    print(f"==>> Qt2: \n{Qt2}")

    t1nn = nn_union[t1]
    t1Qnn = Qaik_rall[t1nn]

    QQ = np.concatenate([Qt1[4:8], Qt2[4:8]], axis=0)
    q1min = QQ[:, 0].min()
    q1max = QQ[:, 0].max()
    q2min = QQ[:, 1].min()
    q2max = QQ[:, 1].max()
    rect = patches.Rectangle(
        (q1min, q2min),
        q1max - q1min,
        q2max - q2min,
        linewidth=1,
        edgecolor="cyan",
        facecolor="none",
        alpha=1,
        label="Bounding Box of IK Solutions",
    )
    ax.add_patch(rect)

    Qaltconfighome = find_alt_config2(qinit, limit2, filterOriginalq=True)

    ax.plot(
        Qaltconfighome[:, 0],
        Qaltconfighome[:, 1],
        "r^",
        markersize=10,
        label="Alt Home Config But not chosen (no teleport)",
    )
    # ax1: C-space
    cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles_extended.npy"))
    ax.plot(
        [qinit[0]],
        [qinit[1]],
        "k-o",
        linewidth=2,
    )
    ax.text(qinit[0], qinit[1], "home config", fontsize=8, ha="right")
    ax.plot(
        cspace_obs[:, 0],
        cspace_obs[:, 1],
        "ro",
        markersize=1,
        label="C-space Obstacles",
        alpha=0.1,
    )
    ax.scatter(
        Qaik[:, :, 0].ravel(),
        Qaik[:, :, 1].ravel(),
        marker="o",
        color="lightgray",
        alpha=0.5,
        label="All IK Solutions",
    )
    # ax.plot(
    #     Qaik_rall[:, :, 0].ravel(),
    #     Qaik_rall[:, :, 1].ravel(),
    #     "gx",
    #     markersize=5,
    #     label="Reachable IK Solutions",
    # )
    # ax.scatter(
    #     Qt1[4:8, 0],
    #     Qt1[4:8, 1],
    #     marker="^",
    #     color="r",
    #     label="t1 Alt IK Solutions",
    # )
    # ax.scatter(
    #     Qt2[4:8, 0],
    #     Qt2[4:8, 1],
    #     marker="^",
    #     color="b",
    #     label="t2 Alt IK Solutions",
    # )
    # cmap = plt.colormaps.get_cmap("tab10")
    # for n in range(len(t1nn)):
    #     ax.scatter(
    #         t1Qnn[n, :, 0],
    #         t1Qnn[n, :, 1],
    #         marker="x",
    #         color=cmap(n),
    #         label="t1 Neighbor IK Solutions",
    #     )
    ax.scatter(
        Qin_threshold[:, 0],
        Qin_threshold[:, 1],
        marker="x",
        color="magenta",
        label="Selected Seed IK Solutions",
    )
    if qtour is not None:
        ax.plot(
            qtour[:, 0],
            qtour[:, 1],
            "m-",
            linewidth=2,
            label="GTSP Tour in C-space",
        )
    ax.set_aspect("equal")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="upper right",
    )
    plt.show()


if __name__ == "__main__":
    # visualize()
    visualize_torus()
