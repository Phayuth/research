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
Qaik = wspace_ik_extended(robot, X)
Qaik_valid = wspace_ik_validity_extended(Qaik, scene)
print(f"==>> X: \n{X}")
print(f"==>> Qaik: \n{Qaik}")
print(f"==>> Qaik_valid: \n{Qaik_valid}")

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

# ------ Estimation of Edges--------------------------------
lmts = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
# estor = RTSPLazyPRMEstimatorExtended(
#     collision_checker=scene.collision_checker,
#     lmts=lmts,
# )
# estor.samples(1000)
estor = RTSPLazyPRMPoissonDiskExtended(
    collision_checker=scene.collision_checker,
    lmts=lmts,
)
estor.samples_sparse(500)
cspace_adjm, store_path, store_cost = RTSP.edgecost_colfree_distance(
    cspace_adjm,
    Q_reachable,
    estor.estimate_shortest_path,
)
print(f"==>> cspace_adjm: \n{cspace_adjm}")

tspace_adjm = RTSP.update_taskspace_adjm(tspace_adjm, cspace_adjm, cluster_ctt)
print(f"==>> tspace_adjm (updated with edge counts): \n{tspace_adjm}")
# ------ End Estimation of Edges-----------------------------

if True:
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

        fig, ax = plt.subplots()
        ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
        ax.plot(qtour[:, 0], qtour[:, 1], "go--", markersize=4, label="GTSP tour")
        for i in range(len(tourid) - 1):
            start_idx = tourid[i]
            end_idx = tourid[i + 1]
            qp = store_path.get((start_idx, end_idx))
            qp = np.array(qp)
            ax.plot(
                qp[:, 0],
                qp[:, 1],
                "b-",
                alpha=0.5,
                label="OMPL path" if i == 0 else None,
            )
            ax.text(
                (qp[0, 0] + qp[-1, 0]) / 2,
                (qp[0, 1] + qp[-1, 1]) / 2 - 0.1,
                f"{start_idx}->{end_idx}",
                color="blue",
                fontsize=8,
            )
            ax.text(
                (qp[0, 0] + qp[-1, 0]) / 2,
                (qp[0, 1] + qp[-1, 1]) / 2,
                f"{store_cost.get((start_idx, end_idx), np.inf):.2f}",
                color="blue",
                fontsize=8,
            )
        ax.set_aspect("equal", "box")
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-2 * np.pi, 2 * np.pi)
        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()
    else:
        print("Tour file not found. Please run GLKH solver file.")
