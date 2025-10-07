import numpy as np
import util
from python_tsp import exact, heuristics
import fast_tsp
import networkx as nx
import time
import pprint


def tspace_distance_position_euclidean(H):
    dists = np.zeros((len(H), len(H)))
    for i in range(10):
        for j in range(10):
            if i != j:
                d = np.linalg.norm(H[i][:3, 3] - H[j][:3, 3])
                dists[i, j] = d
    return dists


def tspace_distance_position_euclidean_orient_geodesic(
    H,
    weight_pos=1.0,
    weight_orient=1.0,
):
    dists = np.zeros(len(H), len(H))
    for i in range(10):
        for j in range(10):
            if i != j:
                dp = np.linalg.norm(H[i][:3, 3] - H[j][:3, 3])
                R1 = H[i][:3, :3]
                R2 = H[j][:3, :3]
                dR = R1.T @ R2
                theta = np.arccos((np.trace(dR) - 1) / 2)
                d = weight_pos * dp + weight_orient * theta
                dists[i, j] = d
    return dists


def tspace_tsp_fast_tsp(dists, method):
    if method == "local_solver":
        tour = fast_tsp.find_tour(dists)
    elif method == "greedy_nearest_neighbor":
        tour = fast_tsp.greedy_nearest_neighbor(dists)
    elif method == "exact_held_karp":
        tour = fast_tsp.solve_tsp_exact(dists)
    cost = fast_tsp.compute_cost(tour, dists)
    return tour, cost


def tspace_tsp_python_tsp(dists, method):
    if method == "exact_brute_force":
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
    length = np.linalg.norm(weights * diffq)
    return length


def cspace_max_euclidean_distance(q1, q2):
    diffq = q2 - q1
    length = np.max(np.abs(diffq))
    return length


def __wraptopi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def cspace_torus_distance(q1, q2):
    q1 = __wraptopi(q1)
    q2 = __wraptopi(q2)
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


def query_collisionfree_path(qa, qb):
    # fake bench time
    for _ in range(1000000):
        1 + 1
    fake_path = np.linspace(qa, qb, num=10)

    # the cost here is real-world robot movement, so we use euclidean
    cost = 0.0
    for i in range(len(fake_path) - 1):
        cost += cspace_euclidean_distance(fake_path[i], fake_path[i + 1])
    return fake_path, cost


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

    def __init__(self, config):
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
        self.log["cspace_collisionfree_tour"] = None
        self.log["cspace_collisionfree_tour_costs"] = None
        self.log["cspace_collisionfree_tour_total_cost"] = None

        self.log["total_solvetime"] = None

    def print_log(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.log)

    def solve(self, H, qinit):
        numsolslist, Qlist = util.solve_ik_bulk(bot, H)
        self.log["cspace_candidate_num"] = sum(numsolslist)
        self.log["cspace_candidate_per_task"] = numsolslist

        Hinit = util.solve_fk(bot, qinit)
        Hnearestid, Hnearest = util.nearest_neighbour_transformation(H, Hinit)

        # 1. Solve tsp tour order in task space
        st = time.time()
        dists_matx = tspace_distance_position_euclidean(H)
        tour, cost = tspace_tsp_python_tsp(
            dists_matx,
            method="exact_dynamic_programming",
        )
        tour = util.rotate_tour_simplifiy_format(tour, Hnearestid)
        et = time.time()
        self.log["tspace_num"] = len(H)
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
        collisionfree_tour, costs = cspace_collisionfree_tour(optimal_config)
        et = time.time()
        self.log["cspace_collisionfree_tour_solvetime"] = et - st
        self.log["cspace_collisionfree_tour"] = collisionfree_tour
        self.log["cspace_collisionfree_tour_costs"] = costs
        self.log["cspace_collisionfree_tour_total_cost"] = sum(costs)

        self.log["total_solvetime"] = (
            self.log["tspace_tour_solvetime"]
            + self.log["cspace_optimal_config_selection_solvetime"]
            + self.log["cspace_collisionfree_tour_solvetime"]
        )


if __name__ == "__main__":
    rtspsolver = RoboTSPSolver(None)
    bot = util.ur5e_dh()
    H = util.generate_random_dh_tasks(bot, 10)
    qinit = np.zeros((6,))

    rtspsolver.solve(H, qinit)
    rtspsolver.print_log()
