import numpy as np
from python_tsp import exact, heuristics
import fast_tsp
import networkx as nx
import time
import pprint
import json
from datetime import datetime
import util as planner_util
from spatial_geometry.utils import Utils

np.set_printoptions(precision=3, suppress=True, linewidth=200)
np.random.seed(42)


def tspace_distance_position_euclidean(H1, H2):
    d = np.linalg.norm(H2[:3, 3] - H1[:3, 3])
    return d


def tspace_distance_matrix_position_euclidean(H):
    dists = np.zeros((len(H), len(H)))
    for i in range(len(H)):
        for j in range(len(H)):
            if i != j:
                d = tspace_distance_position_euclidean(H[j], H[i])
                dists[i, j] = d
    return dists


def tspace_distance_position_euclidean_orient_geodesic(
    H1,
    H2,
    weight_pos=1.0,
    weight_orient=1.0,
):
    dp = np.linalg.norm(H2[:3, 3] - H1[:3, 3])
    R1 = H1[:3, :3]
    R2 = H2[:3, :3]
    dR = R1.T @ R2
    theta = np.arccos((np.trace(dR) - 1) / 2)
    d = weight_pos * dp + weight_orient * theta
    return d


def tspace_distance_matrix_position_euclidean_orient_geodesic(
    H,
    weight_pos=1.0,
    weight_orient=1.0,
):
    dists = np.zeros((len(H), len(H)))
    for i in range(len(H)):
        for j in range(len(H)):
            if i != j:
                d = tspace_distance_position_euclidean_orient_geodesic(
                    H[j],
                    H[i],
                    weight_pos,
                    weight_orient,
                )
                dists[i, j] = d
    return dists


def tspace_tsp_solver(dists, method):
    if method == "local_solver":
        tour = fast_tsp.find_tour(dists)
        cost = fast_tsp.compute_cost(tour, dists)
    elif method == "greedy_nearest_neighbor":
        tour = fast_tsp.greedy_nearest_neighbor(dists)
        cost = fast_tsp.compute_cost(tour, dists)
    elif method == "exact_held_karp":
        tour = fast_tsp.solve_tsp_exact(dists)
        cost = fast_tsp.compute_cost(tour, dists)
    elif method == "exact_brute_force":
        tour, cost = exact.solve_tsp_brute_force(dists)
    elif method == "exact_dynamic_programming":
        tour, cost = exact.solve_tsp_dynamic_programming(dists)
    elif method == "exact_branch_and_bound":
        tour, cost = exact.solve_tsp_branch_and_bound(dists)
    elif method == "heuristic_local_search":
        tour, cost = heuristics.solve_tsp_local_search(dists)
    elif method == "heuristic_simulated_annealing":
        tour, cost = heuristics.solve_tsp_simulated_annealing(dists)
    elif method == "heuristic_lin_kernighan":
        tour, cost = heuristics.solve_tsp_lin_kernighan(dists)
    elif method == "heuristic_record_to_record":
        tour, cost = heuristics.solve_tsp_record_to_record(dists)
    return tour, cost


def cspace_euclidean_distance(q1, q2):
    diffq = q2 - q1
    length = np.linalg.norm(diffq)
    return length


def cspace_weighted_euclidean_distance(q1, q2, weights):
    diffq = q2 - q1
    length = weights * np.linalg.norm(diffq)
    return length


def cspace_max_euclidean_distance(q1, q2):
    diffq = q2 - q1
    length = np.max(np.abs(diffq))
    return length


def _wraptopi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def cspace_torus_distance(q1, q2):
    q1 = _wraptopi(q1)
    q2 = _wraptopi(q2)
    delta = np.abs(q1 - q2)
    delta = np.where(delta > np.pi, 2 * np.pi - delta, delta)
    return np.linalg.norm(delta)


def cspace_candidate_selection_dijkstra(qinit, Qorder, cspace_dist_func):
    Layer = [qinit] + Qorder + [qinit]
    G = nx.DiGraph()
    positions = {}
    node_id = 0
    layers = []
    for lay in Layer:
        lay = np.asarray(lay)
        lid = []
        if lay.ndim == 1:
            positions[node_id] = lay
            G.add_node(node_id)
            lid.append(node_id)
            node_id += 1
        else:
            for l in lay:
                positions[node_id] = l
                G.add_node(node_id)
                lid.append(node_id)
                node_id += 1
        layers.append(lid)

    for i in range(len(layers) - 1):
        for src in layers[i]:
            for dst in layers[i + 1]:
                q1 = positions[src]
                q2 = positions[dst]
                w = cspace_dist_func(q1, q2)
                G.add_edge(src, dst, weight=w)

    config_path_id = nx.dijkstra_path(G, 0, node_id - 1)

    optimal_config = []
    for id in config_path_id:
        optimal_config.append(positions[id])
    optimal_config = np.vstack(optimal_config)
    print(f"==>> optimal_config: \n{optimal_config}")

    cost = 0.0
    for i in range(len(optimal_config) - 1):
        cost += cspace_dist_func(optimal_config[i], optimal_config[i + 1])
    return optimal_config, config_path_id, cost


# def cspace_candidate_selection_dijkstra_multi_sink(
#     qinit, Qorder, cspace_dist_func
# ):
#     limt6 = np.array(
#         [
#             [-2 * np.pi, 2 * np.pi],
#             [-2 * np.pi, 2 * np.pi],
#             [-np.pi, np.pi],
#             [-2 * np.pi, 2 * np.pi],
#             [-2 * np.pi, 2 * np.pi],
#             [-2 * np.pi, 2 * np.pi],
#         ]
#     )
#     Qaltinit = Utils.find_alt_config(qinit.reshape(-1, 1), limt6).T
#     print("Alt init:", Qaltinit)
#     Layer = [qinit] + Qorder + [Qaltinit]
#     G = nx.DiGraph()
#     positions = {}
#     node_id = 0
#     layers = []
#     for lay in Layer:
#         lay = np.asarray(lay)
#         lid = []
#         if lay.ndim == 1:
#             positions[node_id] = lay
#             G.add_node(node_id)
#             lid.append(node_id)
#             node_id += 1
#         else:
#             for l in lay:
#                 positions[node_id] = l
#                 G.add_node(node_id)
#                 lid.append(node_id)
#                 node_id += 1
#         layers.append(lid)

#     for i in range(len(layers) - 1):
#         for src in layers[i]:
#             for dst in layers[i + 1]:
#                 q1 = positions[src]
#                 q2 = positions[dst]
#                 w = cspace_dist_func(q1, q2)
#                 G.add_edge(src, dst, weight=w)

#     last_node_id = node_id
#     target_nodes_id = list(range(last_node_id - len(Qaltinit), last_node_id))
#     print("Target nodes:", target_nodes_id)

#     for tnid in target_nodes_id:
#         print("Target node config:", positions[tnid])

#     config_path_id = None
#     min_cost = float("inf")
#     for tnid in target_nodes_id:
#         try:
#             path_id = nx.dijkstra_path(G, 0, tnid)
#             cost = 0.0
#             for i in range(len(path_id) - 1):
#                 edge_data = G.get_edge_data(path_id[i], path_id[i + 1])
#                 cost += edge_data["weight"]
#             if cost < min_cost:
#                 min_cost = cost
#                 config_path_id = path_id
#         except nx.NetworkXNoPath:
#             continue

#     optimal_config = []
#     for id in config_path_id:
#         optimal_config.append(positions[id])
#     optimal_config = np.vstack(optimal_config)

#     cost = 0.0
#     for i in range(len(optimal_config) - 1):
#         cost += cspace_dist_func(optimal_config[i], optimal_config[i + 1])
#     return optimal_config, config_path_id, cost


def _reroder_taskH(taskH, order):
    taskHreorder = []
    for o in order:
        taskHreorder.append(taskH[o])
    return taskHreorder


def _reorder_Q(Qlist, order):
    Qreorder = []
    for o in order:
        Qreorder.append(Qlist[o])
    return Qreorder


def collision_check_individual(q):
    # fake collision check time
    for _ in range(100000):
        1 + 1
    return np.random.choice([True, False], p=[0.1, 0.9])


def collision_check_Qlist(Qlist, check_func=collision_check_individual):
    QlistCollision = []
    for Q in Qlist:
        QCollision = []
        for q in Q:
            if check_func(q):
                QCollision.append(True)
            else:
                QCollision.append(False)
        QlistCollision.append(QCollision)
    return QlistCollision


def remove_collision_Qlist(Qlist, QlistCollision):
    QlistFree = []
    numQlistFree = []
    for i in range(len(Qlist)):
        Q = Qlist[i]
        QC = QlistCollision[i]
        QFree = []
        for j in range(len(Q)):
            if not QC[j]:
                QFree.append(Q[j])
        numQlistFree.append(len(QFree))
        QlistFree.append(np.array(QFree))
    return QlistFree, numQlistFree


def query_collisionfree_path(qa, qb):
    # fake collision check time
    for _ in range(1000000):
        1 + 1
    path = np.linspace(qa, qb, num=10)

    # the cost here is real-world robot movement, so we use euclidean
    cost = 0.0
    for i in range(len(path) - 1):
        cost += cspace_euclidean_distance(path[i], path[i + 1])
    return path, cost


def cspace_collisionfree_tour(optimal_configs):
    collisionfree_tour = []
    costs = []
    for i in range(len(optimal_configs) - 1):
        qa = optimal_configs[i]
        qb = optimal_configs[i + 1]
        path, cost = query_collisionfree_path(qa, qb)
        collisionfree_tour.append(path)
        costs.append(cost)
    return collisionfree_tour, costs


class RoboTSPSolver:

    def __init__(
        self,
        tspace_dist_matrix_func=tspace_distance_matrix_position_euclidean,
        cspace_dist_func=cspace_euclidean_distance,
        tspace_tsp_solver_func=tspace_tsp_solver,
        tspace_tsp_solver_method="heuristic_lin_kernighan",
        cspace_collisionfree_tour_func=cspace_collisionfree_tour,
        cspace_candidate_selection_func=cspace_candidate_selection_dijkstra,
    ):
        self.tspace_dist_matrix_func = tspace_dist_matrix_func
        self.cspace_dist_func = cspace_dist_func
        self.tspace_tsp_solver_func = tspace_tsp_solver_func
        self.tspace_tsp_solver_method = tspace_tsp_solver_method
        self.cspace_collisionfree_tour_func = cspace_collisionfree_tour_func
        self.cspace_candidate_selection_func = cspace_candidate_selection_func

        self.log = {}
        self.log["tspace_num"] = None
        self.log["tspace_tour"] = None
        self.log["tspace_tour_cost"] = None
        self.log["tspace_tour_solvetime"] = None

        self.log["cspace_candidate_num"] = None
        self.log["cspace_candidate_per_task"] = None
        self.log["cspace_optimal_config_selection_solvetime"] = None
        self.log["cspace_optimal_config_candidate"] = None
        self.log["cspace_optimal_config_cost"] = None

        self.log["cspace_collisionfree_tour_solvetime"] = None
        self.log["cspace_collisionfree_tour_costs"] = None
        self.log["cspace_collisionfree_tour_total_cost"] = None

        self.log["total_solvetime"] = None

    def print_log(self):
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        pp.pprint(self.log)

    def save_log(self):
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Post-process log data for JSON serialization
        processed_log = {}

        for key, value in self.log.items():
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                processed_log[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                # Convert numpy scalars to Python types
                processed_log[key] = value.item()
            elif isinstance(value, list):
                # Handle lists that might contain numpy arrays
                processed_list = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        processed_list.append(item.tolist())
                    elif isinstance(item, (np.integer, np.floating)):
                        processed_list.append(item.item())
                    else:
                        processed_list.append(item)
                processed_log[key] = processed_list
            else:
                # Keep other types as-is (strings, basic numbers, etc.)
                processed_log[key] = value

        # Save as JSON file with timestamp
        json_filename = f"robotsp_solver_log_{timestamp}.json"
        with open(json_filename, "w") as f:
            json.dump(processed_log, f, indent=4)

        print(f"Log saved to: {json_filename}")

        # # Also keep the original pretty-printed text file for human readability
        # with open("robotsp_solver_log.txt", "w") as f:
        #     pp = pprint.PrettyPrinter(indent=4, compact=True, stream=f)
        #     pp.pprint(self.log)

    def solve(self, Htasks, Hinit, qinit, numsolslist, Qlist):
        """
        Htasks: list of task transformation
        Hinit: initial task transformation got from fk(qinit)
        qinit: initial configuration of robot
        numsolslist: list of number of ik solutions per task
        Qlist: list of ik solutions per task
        """
        self.log["cspace_candidate_num"] = sum(numsolslist)
        self.log["cspace_candidate_per_task"] = numsolslist

        # Find nearest task to initial pose to start the tour from there
        Hnearestid, Hnearest = planner_util.nearest_neighbour_H(Htasks, Hinit)

        # 1. Solve tsp tour order in task space
        st = time.time()
        dists_matx = self.tspace_dist_matrix_func(Htasks)
        tour, cost = self.tspace_tsp_solver_func(
            dists_matx,
            method=self.tspace_tsp_solver_method,
        )
        print("OG tour:", tour)
        tour = planner_util.rotate_tour_simplified_format(tour, Hnearestid)
        print("Rotated tour:", tour)
        et = time.time()
        self.log["tspace_num"] = len(Htasks)
        self.log["tspace_tour"] = tour
        self.log["tspace_tour_cost"] = cost
        self.log["tspace_tour_solvetime"] = et - st

        # 2. Solve optimal config in c-space given the order
        Qlist_order = _reorder_Q(Qlist, tour)
        st = time.time()
        optimal_config, config_path_id, cost = (
            self.cspace_candidate_selection_func(
                qinit,
                Qlist_order,
                self.cspace_dist_func,
            )
        )
        et = time.time()
        self.log["cspace_optimal_config_selection_solvetime"] = et - st
        self.log["cspace_optimal_config_candidate"] = optimal_config
        self.log["cspace_optimal_config_cost"] = cost
        self.log["cspace_optimal_config_path_id"] = config_path_id

        # 3. Solve collision-free tour
        st = time.time()
        cf_tour, cf_costs = self.cspace_collisionfree_tour_func(optimal_config)
        et = time.time()
        self.log["cspace_collisionfree_tour_solvetime"] = et - st
        self.log["cspace_collisionfree_tour_costs"] = [float(c) for c in cf_costs]
        self.log["cspace_collisionfree_tour_total_cost"] = sum(cf_costs)

        self.log["total_solvetime"] = (
            self.log["tspace_tour_solvetime"]
            + self.log["cspace_optimal_config_selection_solvetime"]
            + self.log["cspace_collisionfree_tour_solvetime"]
        )

        # self.save_log()
        return cf_tour, cf_costs


if __name__ == "__main__":
    from paper_sequential_planner.experiments.env_planarrr import *
    from paper_sequential_planner.scripts.rtsp_solver import RTSP
    from paper_sequential_planner.scripts.geometric_poses import *

    robot = PlanarRR()
    scene = RobotScene(robot, None)
    rtspsolver = RoboTSPSolver()
    planner = OMPLPlanner(scene.collision_checker)

    # ------- RTSP Preprocessing --------------------------------------
    ntasks = 30
    X = sample_reachable_wspace(ntasks)
    Qaik = wspace_ik(robot, X)
    Qaik_valid = wspace_ik_validity(Qaik, scene)
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
    X_r_full = xlist_to_Xlist(task_reachable)  # (ntasks_rech, 6)
    H_r_full = Xlist_to_Hlist(X_r_full)  # (ntasks_rech, 4, 4)

    Htasks = H_r_full
    # qinit = np.array([1, -1])
    # eepose = np.array(robot.forward_kinematic(qinit))[-1]
    # Hinit = np.eye(4)
    # Hinit[:2, 3] = eepose
    Hinit = H_r_full[0]  # start from the first task reachable pose
    qinit = Q_reachable[0]
    Q_reachable[1] = qinit  # make sure the second config is the same as qinit for testing


    print(num_qreachable)
    print(Q_reachable.shape)
    Qlist = []
    i = 0
    j = 0
    for k in num_qreachable:
        Qlist.append(Q_reachable[i : j + k])
        i = j + k
        j = j + k

    cf_tour, cf_costs = rtspsolver.solve(
        Htasks, Hinit, qinit, num_qreachable, Qlist
    )
    rtspsolver.print_log()

    optimal_task = rtspsolver.log["tspace_tour"]
    optimal_taskH = task_reachable[optimal_task]
    print(f"==>> optimal_taskH: \n{optimal_taskH}")

    optimal_config = rtspsolver.log["cspace_optimal_config_candidate"]
    print(f"==>> optimal_config: \n{optimal_config}")

    tstourid = rtspsolver.log["tspace_tour"]
    print(f"==>> tstourid: \n{tstourid}")
    cstourid = rtspsolver.log["cspace_optimal_config_path_id"]
    print(f"==>> cstourid: \n{cstourid}")

    store_path = {}
    store_cost = {}
    for i in range(optimal_config.shape[0] - 1):
        qa = optimal_config[i]
        qb = optimal_config[i + 1]
        res_ = planner.query_planning(qa, qb)
        if res_ is not None:
            qp, cp = res_
            qp = np.array(qp)
            store_path[i] = qp
            store_cost[i] = cp
        else:
            store_path[i] = None
            store_cost[i] = np.inf
    total_cost = sum(store_cost.get(i, np.inf) for i in range(optimal_config.shape[0] - 1))
    print(f"==>> total_cost: \n{total_cost}")

    def visualize():
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
        ax[0].plot(
            optimal_taskH[:, 0],
            optimal_taskH[:, 1],
            "bo--",
            label="Optimal Taskspace Tour",
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
        ax[1].plot(
            optimal_config[:, 0],
            optimal_config[:, 1],
            linestyle="--",
            color="gray",
            alpha=0.5,
            label="GTSP Tour",
        )
        for i in range(len(store_path)):
            qp = store_path[i]
            if qp is None:
                continue
            ax[1].plot(
                qp[:, 0],
                qp[:, 1],
                "b-",
                alpha=1,
                label="OMPL path" if i == 0 else None,
            )
            ax[1].text((qp[0, 0]), (qp[0, 1]), f"({i})", fontsize=10, ha="right")
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
