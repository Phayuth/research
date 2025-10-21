import numpy as np
from python_tsp import exact, heuristics
import fast_tsp
import networkx as nx
import time
import pprint
import util


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


def tspace_distance_in_cspace_euclidean(H1, H2):
    bot = util.ur5e_dh()
    n1, Q1 = util.solve_ik(bot, H1)
    n2, Q2 = util.solve_ik(bot, H2)
    if n1 == n2:
        diffQ = Q2 - Q1
        dists = np.linalg.norm(diffQ, axis=1)
        return dists
    else:
        print("different number of ik solutions")
        return None


def tspace_distance_in_cspace_torus(H1, H2):
    bot = util.ur5e_dh()
    n1, Q1 = util.solve_ik(bot, H1)
    n2, Q2 = util.solve_ik(bot, H2)
    if n1 == n2:
        dists = np.zeros(n1)
        for i in range(n1):
            q1 = Q1[i]
            q2 = Q2[i]
            dd = cspace_torus_distance(q1, q2)
            dists[i] = dd
        return dists
    else:
        print("different number of ik solutions")
        return None


def tspace_distance_in_cspace_euclidean_dense(H1, H2):
    bot = util.ur5e_dh()
    n1, Q1 = util.solve_ik(bot, H1)
    n2, Q2 = util.solve_ik(bot, H2)
    l = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            q1 = Q1[i]
            q2 = Q2[j]
            diffq = q2 - q1
            length = np.linalg.norm(diffq)
            l[i, j] = length
    return l


def tspace_distance_in_cspace_torus_dense(H1, H2):
    bot = util.ur5e_dh()
    n1, Q1 = util.solve_ik(bot, H1)
    n2, Q2 = util.solve_ik(bot, H2)
    l = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            q1 = Q1[i]
            q2 = Q2[j]
            dd = cspace_torus_distance(q1, q2)
            l[i, j] = dd
    return l


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
        lid = []
        if lay.shape == (6,):
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

    cost = 0.0
    for i in range(len(optimal_config) - 1):
        cost += cspace_dist_func(optimal_config[i], optimal_config[i + 1])
    return optimal_config, cost


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
        tspace_tsp_solver_method="heuristic_local_search",
        cspace_collisionfree_tour_func=cspace_collisionfree_tour,
    ):
        self.tspace_dist_matrix_func = tspace_dist_matrix_func
        self.cspace_dist_func = cspace_dist_func
        self.tspace_tsp_solver_func = tspace_tsp_solver_func
        self.tspace_tsp_solver_method = tspace_tsp_solver_method
        self.cspace_collisionfree_tour = cspace_collisionfree_tour_func

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
        Hnearestid, Hnearest = util.nearest_neighbour_transformation(Htasks, Hinit)

        # 1. Solve tsp tour order in task space
        st = time.time()
        dists_matx = self.tspace_dist_matrix_func(Htasks)
        tour, cost = self.tspace_tsp_solver_func(
            dists_matx,
            method=self.tspace_tsp_solver_method,
        )
        print("OG tour:", tour)
        tour = util.rotate_tour_simplifiy_format(tour, Hnearestid)
        print("Rotated tour:", tour)
        et = time.time()
        self.log["tspace_num"] = len(Htasks)
        self.log["tspace_tour"] = tour
        self.log["tspace_tour_cost"] = cost
        self.log["tspace_tour_solvetime"] = et - st

        # 2. Solve optimal config in c-space given the order
        Qlist_order = _reorder_Q(Qlist, tour)
        st = time.time()
        optimal_config, cost = cspace_candidate_selection_dijkstra(
            qinit,
            Qlist_order,
            cspace_euclidean_distance,
        )
        et = time.time()
        self.log["cspace_optimal_config_selection_solvetime"] = et - st
        self.log["cspace_optimal_config_candidate"] = optimal_config
        self.log["cspace_optimal_config_cost"] = cost

        # 3. Solve collision-free tour
        st = time.time()
        cf_tour, cf_costs = self.cspace_collisionfree_tour(optimal_config)
        et = time.time()
        self.log["cspace_collisionfree_tour_solvetime"] = et - st
        self.log["cspace_collisionfree_tour_costs"] = [float(c) for c in cf_costs]
        self.log["cspace_collisionfree_tour_total_cost"] = sum(cf_costs)

        self.log["total_solvetime"] = (
            self.log["tspace_tour_solvetime"]
            + self.log["cspace_optimal_config_selection_solvetime"]
            + self.log["cspace_collisionfree_tour_solvetime"]
        )

        return cf_tour, cf_costs


if __name__ == "__main__":
    rtspsolver = RoboTSPSolver()

    bot = util.ur5e_dh()
    Htasks = util.generate_random_dh_tasks(bot, 10)
    qinit = np.zeros((6,))
    Hinit = util.solve_fk(bot, qinit)
    numsolslist, Qlist = util.solve_ik_bulk(bot, Htasks)

    # rtspsolver.solve(Htasks, Hinit, qinit, numsolslist, Qlist)
    # rtspsolver.print_log()

    QlistCollision = collision_check_Qlist(Qlist)
    print(QlistCollision)

    QlistFree = remove_collision_Qlist(Qlist, QlistCollision)
    print(QlistFree)

    num, QlistFreeAlt = util.find_altconfig_bulk(QlistFree)
    print(num)
    print(QlistFreeAlt)
