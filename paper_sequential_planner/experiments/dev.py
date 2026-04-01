from paper_sequential_planner.experiments.env_planarrr import *
from paper_sequential_planner.scripts.rtsp_solver import RTSP
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances

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

# distance, on taskspace
tspace_dist = euclidean_distances(X_r)  # (ntasks_rech, ntasks_rech)

# distance, on cspace, by best IK pairing -> shape (ntasks_rech, ntasks_rech, n_ik * altcnf, n_ik * altcnf)
# accessing same id will give us dist to itself, which we dont need.
# always access different task id to get task-to-task distance, which we need
ntasks_rech, n_ik, dof = Qaik_r.shape
_Qflat = Qaik_r.reshape(ntasks_rech * n_ik, dof)
_cspace_dist_flat = nan_euclidean_distances(_Qflat, _Qflat)
cspace_dist = _cspace_dist_flat.reshape(ntasks_rech, n_ik, ntasks_rech, n_ik)
cspace_dist = cspace_dist.transpose(0, 2, 1, 3)

# task-to-task distance by best IK pairing -> shape (ntasks_rech, ntasks_rech)
_cspace_dist_inf = np.where(np.isnan(cspace_dist), np.inf, cspace_dist)
cspace_task_min = _cspace_dist_inf.min(axis=(2, 3))
cspace_task_min[~np.isfinite(cspace_task_min)] = np.nan

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


def visualize():
    fig, ax = plt.subplots(1, 2)

    # example of getting task-to-task distance by best IK pairing
    t1 = 16
    t2 = 19
    print(f"==>> tspace_dist[{t1}, {t2}]: \n{tspace_dist[t1, t2]}")
    print(f"==>> cspace_dist[{t1}, {t2}]: \n{cspace_dist[t1, t2]}")
    print(f"==>> cspace_task_min[{t1}, {t2}]: \n{cspace_task_min[t1, t2]}")

    Xt1 = X_r[t1]
    Xt2 = X_r[t2]
    Qt1r = Qaik_r[t1]
    Qt2r = Qaik_r[t2]

    # ax0: Workspace
    q0 = np.array([1, -1])
    links = np.array(robot.forward_kinematic(q0))

    ax[0].plot(links[:, 0], links[:, 1], "k-", label="Robot at q0")
    ax[0].plot(X[:, 0], X[:, 1], "ro", label="User Input Tasks")
    ax[0].plot(
        task_reachable[:, 0],
        task_reachable[:, 1],
        "gx",
        label="Task-Reachable",
    )
    for i, x in enumerate(X_r):
        ax[0].text(x[0], x[1], f"({i})", fontsize=8, ha="right")
    ax[0].plot([Xt1[0], Xt2[0]], [Xt1[1], Xt2[1]], "c--*", label="Task-to-Task")
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
    )
    ax[1].plot(Qaik[:, :, 0], Qaik[:, :, 1], "bx")
    ax[1].plot(
        Q_reachable[:, 0],
        Q_reachable[:, 1],
        "g*",
        markersize=10,
        label="Q-reachable",
    )
    for q1 in Qt1r:
        for q2 in Qt2r:
            if not np.any(np.isnan(q1)) and not np.any(np.isnan(q2)):
                ax[1].plot(
                    [q1[0], q2[0]],
                    [q1[1], q2[1]],
                    "c--*",
                )
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
