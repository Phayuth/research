from paper_sequential_planner.experiments.env_planarrr import *
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
if ompl_available:
    planner = OMPLPlanner(scene.collision_checker)

ntasks = 30
X = sample_reachable_wspace(ntasks)  # (ntasks, 2)
Qaik = wspace_ik_extended(robot, X)  # (ntasks, n_ik * altcnf, dof)
Qaik_valid = wspace_ik_validity_extended(Qaik, scene)  # (ntasks, n_ik * altcnf, 1)

# filterout reachable/unreachable tasks
X_isunr = np.all(Qaik_valid != 1, axis=1).flatten()  # (ntasks, ) True/False
X_r = X[~X_isunr]  # (ntasks_rech, 2)
Qaik_valid_r = Qaik_valid[~X_isunr]  # (ntasks_rech, n_ik * altcnf, 1)
Qaik_r = Qaik[~X_isunr]  # (ntasks_rech, n_ik * altcnf, dof)
Qaik_r = np.where(Qaik_valid_r == 1, Qaik_r, np.nan)  # set value to nan if invalid
print(f"==>> Qaik_r.shape: \n{Qaik_r.shape}")

# taskspace distance
X_r_full = xlist_to_Xlist(X_r)  # (ntasks_rech, 6)
H_r_full = Xlist_to_Hlist(X_r_full)  # (ntasks_rech, 4, 4)
tspace_dist = se3_error_pairwise_distance(H_r_full, 0.2)  # (ntasks_r, ntasks_r)
print(f"==>> tspace_dist.shape: \n{tspace_dist.shape}")

# cspace distance by IK pairing
# shape (ntasks_rech, ntasks_rech, n_ik * altcnf, n_ik * altcnf)
# accessing same id will give us dist to itself, which we dont need.
# always access different task id to get task-to-task distance, which we need
ntasks_rech, n_ik, dof = Qaik_r.shape
_Qflat = Qaik_r.reshape(ntasks_rech * n_ik, dof)
_cspace_dist_flat = nan_euclidean_distances(_Qflat, _Qflat)
cspace_dist = _cspace_dist_flat.reshape(ntasks_rech, n_ik, ntasks_rech, n_ik)
cspace_dist = cspace_dist.transpose(0, 2, 1, 3)
print(f"==>> cspace_dist.shape: \n{cspace_dist.shape}")

# task-to-task distance by best IK pairing -> shape (ntasks_rech, ntasks_rech)
_cspace_dist_inf = np.where(np.isnan(cspace_dist), np.inf, cspace_dist)
cspace_task_min = _cspace_dist_inf.min(axis=(2, 3))
cspace_task_min[~np.isfinite(cspace_task_min)] = np.nan
print(f"==>> cspace_task_min.shape: \n{cspace_task_min.shape}")

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

# getting data
t1 = 1
t2 = 18
Qaik_r_t1 = Qaik_r[t1]  # (n_ik, dof)
Qaik_r_t2 = Qaik_r[t2]  # (n_ik, dof)
print(f"==>> Qaik_r_t1: \n{Qaik_r_t1}")
print(f"==>> Qaik_r_t2: \n{Qaik_r_t2}")
# format of cspace_dist
# [[t1q1-t2q1, t1q1-t2q2]]
# [[t2q1-t1q1, t2q1-t1q2]]
t1t2_cspace_dist = cspace_dist[t1, t2]  # (n_ik, n_ik)
print(f"==>> t1t2_cspace_dist: \n{t1t2_cspace_dist}")

# unique pairs of task-to-task = ik_num*altc_num*3^dof
group_matrix = np.zeros_like(t1t2_cspace_dist, dtype=int)
ik_num = 2  # elbow up and down
altc_num = 4  # 4 alt config per ik solution
for i in range(ik_num):
    for j in range(ik_num):
        # DIST = t1t2_cspace_dist[
        #     i * altc_num : i * altc_num + altc_num,
        #     j * altc_num : j * altc_num + altc_num,
        # ]
        group_pairs, groups_num, total_pairs, groups_matrix = (
            find_altconfig_redudancy(
                Qaik_r_t1[i * altc_num : i * altc_num + altc_num],
                Qaik_r_t2[j * altc_num : j * altc_num + altc_num],
            )
        )
        group_matrix[
            i * altc_num : i * altc_num + altc_num,
            j * altc_num : j * altc_num + altc_num,
        ] = groups_matrix
print(f"==>> group_matrix: \n{group_matrix}")


def task_to_task_configuration_interp(Qaik_r, nintp=10):
    """
    Interpolate between all pairs of task-reachable configurations,
    and set to nan if either of the pair is invalid.
    final shape (ntasks_rech, ntasks_rech, n_ik, n_ik, nintp, dof)
    Ex:
    t0t1q0q0 = interp[0, 1, 0, 0]
    give us interp bet/ first IK of task0 to first IK of task1.
    task id should not be the same
    """
    ntasks_rech, n_ik, dof = Qaik_r.shape
    Q1 = Qaik_r[:, None, :, None, None, :]  # (ntasks_rech,1,n_ik,1,1,dof)
    Q2 = Qaik_r[None, :, None, :, None, :]  # (1,ntasks_rech,1,n_ik,1,dof)
    tau = np.linspace(0.0, 1.0, nintp, dtype=Qaik_r.dtype)
    tau = tau[None, None, None, None, :, None]  # (1,1,1,1,nintp,1)
    # (ntasks_rech,ntasks_rech,n_ik,n_ik,nintp,dof)
    interp = (1.0 - tau) * Q1 + tau * Q2
    invalid_cfg = np.isnan(Qaik_r).all(axis=-1)  # (ntasks_rech,n_ik)
    # (ntasks_rech,ntasks_rech,n_ik,n_ik)
    invalid_pair = invalid_cfg[:, None, :, None] | invalid_cfg[None, :, None, :]
    interp = np.where(invalid_pair[..., None, None], np.nan, interp)
    return interp


tt_cspace_interp = task_to_task_configuration_interp(Qaik_r, nintp=10)
print(f"==>> tt_cspace_interp.shape: \n{tt_cspace_interp.shape}")
t0t1q0q0 = tt_cspace_interp[0, 1, 0, 0]


(
    task_reachable,
    num_treachable,
    num_qreachable,
    Q_reachable,
    cluster_ttc,
    cluster_ctt,
    tspace_adjm,
    cspace_adjm,
) = RTSP.preprocess(X, Qaik, Qaik_valid)
# print(f"==>> task_reachable: \n{task_reachable}")
# print(f"==>> num_treachable: \n{num_treachable}")
# print(f"==>> num_qreachable: \n{num_qreachable}")
# print(f"==>> Q_reachable: \n{Q_reachable}")
# print(f"==>> cluster_ttc: \n{cluster_ttc}")
# print(f"==>> cluster_ctt: \n{cluster_ctt}")
# print(f"==>> tspace_adjm: \n{tspace_adjm}")
# print(f"==>> cspace_adjm: \n{cspace_adjm}")


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


nnr = 0.5
nnk = 5
nn_r = radius_neighbors(tspace_dist, radius=nnr)
print(f"==>> nn_r: \n{nn_r}")
nn_k = knn_from_distance(tspace_dist, k=nnk)
print(f"==>> nn_k: \n{nn_k}")

nn_union = []
for i in range(tspace_dist.shape[0]):
    union_set = set(nn_r[i]) | set(nn_k[i])
    nn_union.append(sorted(union_set))
print(f"==>> nn_union: \n{nn_union}")

# contruct taskspace graph connection adjmat from nnunion
task_to_task_adj = np.zeros_like(tspace_dist, dtype=bool)
for i in range(tspace_dist.shape[0]):
    for j in nn_union[i]:
        task_to_task_adj[i, j] = True
        task_to_task_adj[j, i] = True  # undirected
print(f"==>> task_to_task_adj: \n{task_to_task_adj}")

# from this adjmat, we eliminate the task-to-task q pairs

# raise
def visualize():
    fig, ax = plt.subplots(1, 2)

    # example of getting task-to-task distance by best IK pairing
    # neighboring tasks in task space
    t1 = 1
    t2 = 18
    Xt1 = X_r[t1]
    Xt2 = X_r[t2]
    Qt1r = Qaik_r[t1]
    Qt2r = Qaik_r[t2]
    neart1 = nn_r[t1]
    neart2 = nn_r[t2]
    Xneart1 = X_r[neart1]
    Xneart2 = X_r[neart2]
    print(f"==>> neart1: \n{neart1}")
    print(f"==>> neart2: \n{neart2}")
    print(f"==>> tspace_dist[{t1}, {t2}]: \n{tspace_dist[t1, t2]}")
    print(f"==>> cspace_dist[{t1}, {t2}]: \n{cspace_dist[t1, t2]}")
    print(f"==>> cspace_task_min[{t1}, {t2}]: \n{cspace_task_min[t1, t2]}")

    # obstacles
    for shp in scene.obstacles:
        x, y = shp.exterior.xy
        ax[0].fill(x, y, alpha=0.5, fc="red", ec="black")

    cirt1 = plt.Circle(
        Xt1, nnr, color="r", fill=False, linestyle="--", label="t1 radius"
    )
    cirt2 = plt.Circle(
        Xt2, nnr, color="r", fill=False, linestyle="--", label="t2 radius"
    )
    ax[0].add_artist(cirt1)
    ax[0].add_artist(cirt2)

    nnkt1 = nn_k[t1]
    nnrt1 = nn_r[t1]
    for nnt1 in nnkt1:
        ax[0].plot(
            [Xt1[0], X_r[nnt1][0]],
            [Xt1[1], X_r[nnt1][1]],
            "b--",
            label="t1 knn" if nnt1 == nnkt1[0] else "",
        )
    for nnt1 in nnrt1:
        ax[0].plot(
            [Xt1[0], X_r[nnt1][0]],
            [Xt1[1], X_r[nnt1][1]],
            "r--",
            label="t1 radius" if nnt1 == nnrt1[0] else "",
        )

    # ax0: Workspace
    q0 = np.array([1, -1])
    links = np.array(robot.forward_kinematic(q0))

    ax[0].plot(links[:, 0], links[:, 1], "k-o", linewidth=2, label="Robot at q0")
    ax[0].plot(
        X[:, 0],
        X[:, 1],
        "o",
        color="lightgray",
        label="User Input Tasks",
    )
    ax[0].plot(
        task_reachable[:, 0],
        task_reachable[:, 1],
        "gx",
        label="Task-Reachable",
    )
    for i, x in enumerate(X_r):
        ax[0].text(x[0], x[1], f"({i})", fontsize=8, ha="right")
    ax[0].plot([Xt1[0], Xt2[0]], [Xt1[1], Xt2[1]], "c--", label="Task-to-Task")
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
        Q_reachable[:, 0],
        Q_reachable[:, 1],
        "gx",
        markersize=5,
        label="Q-reachable",
    )
    for q1 in Qt1r:
        for q2 in Qt2r:
            if not np.any(np.isnan(q1)) and not np.any(np.isnan(q2)):
                ax[1].plot(
                    [q1[0], q2[0]],
                    [q1[1], q2[1]],
                    "c--",
                    # remove duplicate legend by only labeling the first valid pair
                    label=(
                        "Task-to-Task by IK"
                        if "Task-to-Task by IK"
                        not in ax[1].get_legend_handles_labels()[1]
                        else ""
                    ),
                )
    ax[1].plot(Qt1r[:, 0], Qt1r[:, 1], "ro", label="t1 IK Solutions")
    ax[1].plot(Qt2r[:, 0], Qt2r[:, 1], "bo", label="t2 IK Solutions")
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
