from paper_sequential_planner.experiments.env_planarrr import *
from paper_sequential_planner.scripts.rtsp_solver import RTSP, GLKHHelper
from paper_sequential_planner.scripts.rtsp_lazyprm import (
    RTSPLazyPRMEstimatorExtended,
    RTSPLazyPRMPoissonDiskExtended,
)

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robot = PlanarRR()
scene = RobotScene(robot, None)
if ompl_available:
    planner = OMPLPlanner(scene.collision_checker, limit=2 * np.pi)

# ------- RTSP Preprocessing --------------------------------------
ntasks = 30
X = sample_reachable_wspace(ntasks)
qinit = np.array([1, -1])
Xinit = np.array([robot.forward_kinematic(qinit)[-1]])
Qaik = wspace_ik_extended(robot, X)
Qaik_valid = wspace_ik_validity_extended(Qaik, scene)

# concate Xinit, qinit
# X = np.vstack((Xinit, X))
# Qaikinit = np.full((1, Qaik.shape[1], Qaik.shape[2]), qinit)
# Qaikinit_valid = np.full((1, Qaik_valid.shape[1], Qaik_valid.shape[2]), 1)
# Qaik = np.vstack((Qaikinit, Qaik))
# Qaik_valid = np.vstack((Qaikinit_valid, Qaik_valid))


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

num_unique_edges = RTSP.num_edges_unique(num_qreachable)
print(f"==>> num_unique_edges: \n{num_unique_edges}")
num_supercluster_edges = RTSP.num_supercluster_edges(num_treachable)
print(f"==>> num_supercluster_edges: \n{num_supercluster_edges}")
# ------- End RTSP Preprocessing --------------------------------------

# ------- Compute Initial Cost --------------------------------------
cspace_adjm_euc_min = RTSP.edgecost_eucl_distance(Q_reachable, cspace_adjm)
print(f"==>> cspace_adjm_euc_min: \n{cspace_adjm_euc_min}")
# ------- End Compute Initial Cost ----------------------------------

# ------- Queuing System ------------------------------------
# Queue priority edges to be estimated first
for i in range(num_treachable):
    for j in range(i + 1, num_treachable):
        cpij, idpij = RTSP.get_cost_task_to_task(
            cluster_ttc, cspace_adjm_euc_min, i, j
        )
        cpsortidx = np.argsort(cpij)
        cpsort = [cpij[i] for i in cpsortidx]
        idpsort = [idpij[i] for i in cpsortidx]
        print(f"Sorted t{i}->{j}: costs: {cpsort}, id pairs: {idpsort}")
# ------- End Queuing System --------------------------------

# # ------ Estimation of Edges--------------------------------
# lmts = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
# # estor = RTSPLazyPRMEstimatorExtended(
# #     collision_checker=scene.collision_checker,
# #     lmts=lmts,
# # )
# # estor.samples(1000)
# estor = RTSPLazyPRMPoissonDiskExtended(
#     collision_checker=scene.collision_checker,
#     lmts=lmts,
# )
# estor.samples_sparse(500)
# cspace_adjm, store_path, store_cost = RTSP.edgecost_colfree_distance(
#     cspace_adjm,
#     Q_reachable,
#     estor.estimate_shortest_path,
# )
# print(f"==>> cspace_adjm: \n{cspace_adjm}")

# tspace_adjm = RTSP.update_taskspace_adjm(tspace_adjm, cspace_adjm, cluster_ctt)
# print(f"==>> tspace_adjm (updated with edge counts): \n{tspace_adjm}")
# # ------ End Estimation of Edges-----------------------------

if False:
    # plot debug cspace
    cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles_extended.npy"))
    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
    ax.plot(Q_reachable[:, 0], Q_reachable[:, 1], "g*", markersize=10)
    for i, path in enumerate(store_path.values()):
        ax.plot(
            path[:, 0],
            path[:, 1],
            linewidth=2,
        )
    ax.set_aspect("equal", "box")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.grid(True)
    plt.show()

if True:
    # write GTSP problem file for GLKH
    GLKHHelper.write_glkh_fullmatrix_file(
        os.path.join(GLKHHelper.problemdir, "problem_planarrr_ext.gtsp"),
        cspace_adjm,
        cluster_ttc,
    )

    # solve GTSP using GLKH
    if os.path.exists(
        os.path.join(GLKHHelper.problemdir, "problem_planarrr_ext.tour")
    ):
        tourid = GLKHHelper.read_tour_file(
            os.path.join(GLKHHelper.problemdir, "problem_planarrr_ext.tour")
        )
        print(f"==>> tourid: \n{tourid}")
        cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles_extended.npy"))
        qtour, store_path, store_cost = RTSP.postprocess(
            tourid, Q_reachable, planner.query_planning
        )
        total_cost = sum(store_cost.get((tourid[i], tourid[i + 1]), np.inf) for i in range(len(tourid) - 1))
        print(f"==>> qtour: \n{qtour}")
        print(f"==>> total_cost: \n{total_cost}")

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f"cost {total_cost:.3f}")

        # obstacles
        for shp in scene.obstacles:
            x, y = shp.exterior.xy
            ax[0].fill(x, y, alpha=0.5, fc="red", ec="black")

        # ax0: Workspace
        links = np.array(robot.forward_kinematic(qinit))

        ax[0].plot(
            links[:, 0], links[:, 1], "k-o", linewidth=2, label="Robot at qinit"
        )
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
        for i, x in enumerate(task_reachable):
            ax[0].text(x[0], x[1], f"({i})", fontsize=10, ha="right")
        ax[0].set_aspect("equal")
        ax[0].set_xlim(-4, 4)
        ax[0].set_ylim(-4, 4)
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")
        ax[0].legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
        )

        ax[1].plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
        ax[1].plot(qtour[:, 0], qtour[:, 1], "go--", markersize=4, label="GTSP tour")
        for i in range(len(tourid) - 1):
            start_idx = tourid[i]
            end_idx = tourid[i + 1]
            qp = store_path.get((start_idx, end_idx))
            qp = np.array(qp)
            ax[1].plot(
                qp[:, 0],
                qp[:, 1],
                "b-",
                alpha=0.5,
                label="OMPL path" if i == 0 else None,
            )
            ax[1].text(
                (qp[0, 0] + qp[-1, 0]) / 2,
                (qp[0, 1] + qp[-1, 1]) / 2 - 0.1,
                f"{start_idx}->{end_idx}",
                color="blue",
                fontsize=8,
            )
            ax[1].text(
                (qp[0, 0] + qp[-1, 0]) / 2,
                (qp[0, 1] + qp[-1, 1]) / 2,
                f"{store_cost.get((start_idx, end_idx), np.inf):.2f}",
                color="blue",
                fontsize=8,
            )
        ax[1].set_aspect("equal", "box")
        ax[1].set_xlim(-2 * np.pi, 2 * np.pi)
        ax[1].set_ylim(-2 * np.pi, 2 * np.pi)
        ax[1].grid(True)
        ax[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()
    else:
        print("Tour file not found. Please run GLKH solver file.")
