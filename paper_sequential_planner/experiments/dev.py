from paper_sequential_planner.experiments.env_planarrr import *
from paper_sequential_planner.scripts.rtsp_solver import RTSP
from paper_sequential_planner.scripts.geometric_Xmean import fit
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robot = PlanarRR()
scene = RobotScene(robot, None)
if ompl_available:
    planner = OMPLPlanner(scene.collision_checker)

ntasks = 10
X = sample_reachable_wspace(ntasks)
Qaik = wspace_ik_extended(robot, X)
Qaik_valid = wspace_ik_validity_extended(Qaik, scene)
print(f"==>> X: \n{X}")
print(f"==>> Qaik: \n{Qaik}")
print(f"==>> Qaik_valid: \n{Qaik_valid}")

# Qaik_filtered = np.where(Qaik_valid == 1, Qaik, np.nan)

# remove the task that ik are all invalid
is_unreachable = np.all(Qaik_valid != 1, axis=1).flatten()
print(f"==>> is_unreachable: \n{is_unreachable}")

Xreachable = X[~is_unreachable]
print(f"==>> Xreachable: \n{Xreachable}")
Qreachable = Qaik[~is_unreachable]
print(f"==>> Qreachable: \n{Qreachable}")
Qaik_valid_filtered = Qaik_valid[~is_unreachable]
print(f"==>> Qaik_valid_filtered: \n{Qaik_valid_filtered}")
Qreachable = np.where(Qaik_valid_filtered == 1, Qreachable, np.nan)
print(f"==>> Qreachable.shape: \n{Qreachable.shape}")
print(f"==>> Qreachable (after filtering): \n{Qreachable}")


tspace_dist = euclidean_distances(Xreachable)
print(f"==>> tspace_dist.shape: \n{tspace_dist.shape}")
print(f"==>> tspace_dist: \n{tspace_dist}")

t12_dist = tspace_dist[0, 1]
print(f"==>> t12_dist: \n{t12_dist}")

t1Q = Qreachable[0]
print(f"==>> t1Q: \n{t1Q}")
t2Q = Qreachable[1]
print(f"==>> t2Q: \n{t2Q}")
# Distance between IK sets of two tasks -> shape (8, 8)
cspace_dist_withnan = nan_euclidean_distances(t1Q, t2Q)
print(f"==>> cspace_dist_withnan.shape: \n{cspace_dist_withnan.shape}")
print(f"==>> cspace_dist_withnan: \n{cspace_dist_withnan}")


# Keep matrix structure: (n_task, n_task, n_ik, n_ik)
n_task, n_ik, dof = Qreachable.shape
Qflat = Qreachable.reshape(n_task * n_ik, dof)
cspace_dist_flat = nan_euclidean_distances(Qflat, Qflat)
cspace_dist_all = cspace_dist_flat.reshape(n_task, n_ik, n_task, n_ik).transpose(
    0, 2, 1, 3
)
print(
    f"==>> cspace_dist_all.shape (task, task, ik, ik): \n{cspace_dist_all.shape}"
)

# accessing same id will give us dist to itself, which we dont need.
# always access different task id to get task-to-task distance, which we need
t1t2 = cspace_dist_all[0, 1]
print(f"==>> t1t2: \n{t1t2}")
mint1t2 = np.nanmin(t1t2)
print(f"==>> mint1t2: \n{mint1t2}")
argmint1t2 = np.nanargmin(t1t2)
print(f"==>> argmint1t2: \n{argmint1t2}")

# Optional task-to-task distance by best IK pairing -> shape (n_task, n_task)
cspace_dist_all_safe = np.where(np.isnan(cspace_dist_all), np.inf, cspace_dist_all)
cspace_task_min = cspace_dist_all_safe.min(axis=(2, 3))
cspace_task_min[~np.isfinite(cspace_task_min)] = np.nan
print(f"==>> cspace_task_min.shape: \n{cspace_task_min.shape}")
print(f"==>> cspace_task_min: \n{cspace_task_min}")


raise

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
print(f"==>> task_reachable: \n{task_reachable}")
print(f"==>> num_treachable: \n{num_treachable}")
print(f"==>> num_qreachable: \n{num_qreachable}")
print(f"==>> Q_reachable: \n{Q_reachable}")
print(f"==>> cluster_ttc: \n{cluster_ttc}")
print(f"==>> cluster_ctt: \n{cluster_ctt}")
print(f"==>> tspace_adjm: \n{tspace_adjm}")
print(f"==>> cspace_adjm: \n{cspace_adjm}")

raise

# here we want to develop a cluster of task to reduce num edges
# cluster cost metric is to be discussed later
# we want to identify task clusters that are close, meaning it learn
# they belong to the same topological region
dof = 2
kmin = 5
kmax = 40
weights = [0.015] * dof
xmeans = fit(
    task_reachable,
    kmax=kmax,
    kmin=kmin,
    weights=np.array(weights) * dof,
)

N = xmeans.k
print(f"==>> N: \n{N}")
labels = xmeans.labels_
print(f"==>> labels: \n{labels}")
center = xmeans.centroid_centres_
points_per_cluster = xmeans.count
print(f"==>> points_per_cluster: \n{points_per_cluster}")

#


raise


def visualize():
    fig, ax = plt.subplots(1, 2)

    # ax0: Workspace
    q0 = np.array([1, -1])
    links = np.array(robot.forward_kinematic(q0))
    ax[0].plot(links[:, 0], links[:, 1], "k-", label="Robot at q0")
    ax[0].plot(X[:, 0], X[:, 1], "ro", label="Reachable Workspace")
    ax[0].plot(
        task_reachable[:, 0],
        task_reachable[:, 1],
        "gx",
        label="Task-Reachable",
    )
    ax[0].scatter(
        task_reachable[:, 0],
        task_reachable[:, 1],
        c=labels,
        # cmap="tab20",
        s=100,
        edgecolors="k",
        label="Clusters",
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
