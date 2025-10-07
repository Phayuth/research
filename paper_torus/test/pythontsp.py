import fast_tsp
from python_tsp import exact, heuristics
import numpy as np
from sklearn.metrics import pairwise_distances

dists = np.array(
    [
        [0, 63, 72, 70],
        [63, 0, 57, 53],
        [72, 57, 0, 4],
        [70, 53, 4, 0],
    ]
)


def fast_tsppp():
    # Run a local solver to find a near-optimal TSP tour. For small problems, the exact solution is returned.
    tour = fast_tsp.find_tour(dists)
    print(tour)

    cost_solver = fast_tsp.compute_cost(tour, dists)
    print(cost_solver)

    # Solve TSP using the greedy nearest neighbor heuristic. This is a very rough approximation to the exact solution. The preferred way is to use find_tour.
    tour_greedy = fast_tsp.greedy_nearest_neighbor(dists)
    print(tour_greedy)

    cc = fast_tsp.compute_cost(tour_greedy, dists)
    print(cc)

    # Solves TSP optimally using the bottom-up Held-Karp’s algorithm in time. This is tractable for small n but quickly becomes untractable for n even of medium size.
    tour_exact = fast_tsp.solve_tsp_exact(dists)
    print(tour_exact)

    cost_exact = fast_tsp.compute_cost(tour_exact, dists)
    print(cost_exact)


def python_tsppp():
    permutation, distance = exact.solve_tsp_dynamic_programming(dists)
    print(permutation)
    print(distance)

    permutation2, distance2 = exact.solve_tsp_brute_force(dists)
    print(permutation2)
    print(distance2)

    permutation3, distance3 = exact.solve_tsp_branch_and_bound(dists)
    print(permutation3)
    print(distance3)

    permutation3, distance3 = heuristics.solve_tsp_local_search(dists)
    print(permutation3)
    print(distance3)

    permutation4, distance4 = heuristics.solve_tsp_simulated_annealing(dists)
    print(permutation4)
    print(distance4)

    permutation5, distance5 = heuristics.solve_tsp_lin_kernighan(dists)
    print(permutation5)
    print(distance5)

    permutation6, distance6 = heuristics.solve_tsp_record_to_record(dists)
    print(permutation6)
    print(distance6)


if __name__ == "__main__":
    # fast_tsppp()
    python_tsppp()
