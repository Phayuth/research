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

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robot = PlanarRR()
scene = RobotScene(robot, None)
planner = OMPLPlanner(scene.collision_checker)


# initial config and task  -----------------------------------------
qinit = np.array([1, -1])
Xinit = np.array([robot.forward_kinematic(qinit)[-1]])
# -----------------------------------------------------------------

# to visit task ---------------------------------------------------
ntasks = 30
X = sample_reachable_wspace(ntasks)  # (ntasks, 2)
Qaik = wspace_ik_extended(robot, X)  # (ntasks, n_ik * altcnf, dof)
Qaik_valid = wspace_ik_validity_extended(Qaik, scene)  # (ntasks, n_ik * altcnf, 1)
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
    t0t1q0q0 = interp[0, 1, 0, 0]
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


tt_cspace_interp = task_to_task_configuration_interp(Qaik_rall, nintp=10)
print(f"==>> tt_cspace_interp.shape: \n{tt_cspace_interp.shape}")
t0t1q0q0 = tt_cspace_interp[0, 1, 0, 0]


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
nnr = 0.5
nnk = 5
nn_r = radius_neighbors(tspace_dist, radius=nnr)
nn_k = knn_from_distance(tspace_dist, k=nnk)
nn_union = []
for i in range(tspace_dist.shape[0]):
    union_set = set(nn_r[i]) | set(nn_k[i])
    nn_union.append(sorted(union_set))
print(f"==>> nn_union: \n{nn_union}")
nn_dist = []
for i in range(len(nn_union)):
    dists = [tspace_dist[i, j].item() for j in nn_union[i]]
    nn_dist.append(dists)
print(f"==>> nn_dist: \n{nn_dist}")
# -----------------------------------------------------------------


# contruct taskspace graph connection adjmat from nnunion ------------
# row (from), col (to), value (connected or not)
task_to_task_adj = np.zeros_like(tspace_dist, dtype=bool)
for i in range(tspace_dist.shape[0]):
    for j in nn_union[i]:
        task_to_task_adj[i, j] = True
        task_to_task_adj[j, i] = True  # undirected
print(f"==>> task_to_task_adj.shape: \n{task_to_task_adj.shape}")
print(f"==>> task_to_task_adj: \n{task_to_task_adj}")

# from this adjmat, we eliminate the task-to-task q pairs
c_to_c_adj = np.zeros_like(cspace_eudist, dtype=bool)
for i in range(task_to_task_adj.shape[0]):
    for j in range(i + 1, task_to_task_adj.shape[0]):
        if task_to_task_adj[i, j]:
            c_to_c_adj[i, j] = True
            c_to_c_adj[j, i] = True  # undirected
print(f"==>> c_to_c_adj.shape: \n{c_to_c_adj.shape}")
# -----------------------------------------------------------------

# unique groups of IK ----------------------------------------------
ik_num = 2  # elbow up and down
altc_num = 4  # 4 alt config per ik solution
group_mat = np.zeros_like(c_to_c_adj, dtype=int)
for ti in range(task_to_task_adj.shape[0]):
    for tj in range(ti + 1, task_to_task_adj.shape[0]):
        if task_to_task_adj[ti, tj]:
            for ik_i in range(n_ik):
                for ik_j in range(n_ik):
                    DIST = cspace_eudist[
                        ti,
                        tj,
                        ik_i * altc_num : ik_i * altc_num + altc_num,
                        ik_j * altc_num : ik_j * altc_num + altc_num,
                    ]
                    if np.isnan(DIST).all():
                        group_mat[
                            ti,
                            tj,
                            ik_i * altc_num : ik_i * altc_num + altc_num,
                            ik_j * altc_num : ik_j * altc_num + altc_num,
                        ] = -1
                    else:
                        group_pairs, groups_num, total_pairs, groups_matrix = (
                            find_altconfig_redudancy(
                                Qaik_rall[
                                    ti,
                                    ik_i * altc_num : ik_i * altc_num + altc_num,
                                ],
                                Qaik_rall[
                                    tj,
                                    ik_j * altc_num : ik_j * altc_num + altc_num,
                                ],
                                DIST,
                            )
                        )
                        group_mat[
                            ti,
                            tj,
                            ik_i * altc_num : ik_i * altc_num + altc_num,
                            ik_j * altc_num : ik_j * altc_num + altc_num,
                        ] = groups_matrix
print(f"==>> group_mat.shape: \n{group_mat.shape}")
# -----------------------------------------------------------------
t1 = 1
# t1t2_gm = group_mat[t1, t2]
# print(f"==>> t1t2_gm: \n{t1t2_gm}")

t1nn = nn_union[t1]
print(f"==>> t1nn: \n{t1nn}")
t1nn_cspace_eudist = cspace_eudist[t1, t1nn]  # (n_nn, n_ik, n_nn, n_ik)
print(f"==>> t1nn_cspace_eudist: \n{t1nn_cspace_eudist}")
print(f"==>> t1nn_cspace_eudist.shape: \n{t1nn_cspace_eudist.shape}")


minvalue = np.nanmin(t1nn_cspace_eudist)
print(f"==>> minvalue: \n{minvalue}")
maxvalue = np.nanmax(t1nn_cspace_eudist)
print(f"==>> maxvalue: \n{maxvalue}")

epslGH = 1.0
ddd= t1nn_cspace_eudist

raise


def visualize():
    fig, ax = plt.subplots(1, 2)

    # example of getting task-to-task distance by best IK pairing
    # neighboring tasks in task space
    t1 = 5
    Xt1 = X_rall[t1]

    # obstacles
    for shp in scene.obstacles:
        x, y = shp.exterior.xy
        ax[0].fill(x, y, alpha=0.5, fc="red", ec="black")

    cirt1 = plt.Circle(
        Xt1, nnr, color="r", fill=False, linestyle="--", label="t1 radius"
    )
    ax[0].add_artist(cirt1)

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
    for i, x in enumerate(X_rall):
        ax[0].text(x[0], x[1], f"({i})", fontsize=8, ha="right")
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-4, 4)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
    )

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
    # for q1 in Qt1r:
    #     for q2 in Qt2r:
    #         if not np.any(np.isnan(q1)) and not np.any(np.isnan(q2)):
    #             ax[1].plot(
    #                 [q1[0], q2[0]],
    #                 [q1[1], q2[1]],
    #                 "c--",
    #                 # remove duplicate legend by only labeling the first valid pair
    #                 label=(
    #                     "Task-to-Task by IK"
    #                     if "Task-to-Task by IK"
    #                     not in ax[1].get_legend_handles_labels()[1]
    #                     else ""
    #                 ),
    #             )
    # ax[1].plot(Qt1r[:, 0], Qt1r[:, 1], "ro", label="t1 IK Solutions")
    # ax[1].plot(Qt2r[:, 0], Qt2r[:, 1], "bo", label="t2 IK Solutions")
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


visualize()
