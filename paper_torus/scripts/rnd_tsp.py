import numpy as np
import pyomo.environ as pyo


def solve_tsp_mip_mtz(coords):
    cities = list(coords.keys())

    # Compute Euclidean distances
    def euclidean(i, j):
        return np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))

    distances = {(i, j): euclidean(i, j) for i in cities for j in cities if i != j}

    #  Pyomo Model
    model = pyo.ConcreteModel()
    model.N = pyo.Set(initialize=cities)
    model.x = pyo.Var(model.N, model.N, domain=pyo.Binary)
    model.u = pyo.Var(
        model.N, domain=pyo.NonNegativeReals, bounds=(0, len(cities) - 1)
    )

    # Objective: minimize total distance
    def obj_rule(model):
        return sum(
            distances[i, j] * model.x[i, j]
            for i in model.N
            for j in model.N
            if i != j
        )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints
    def no_self_loop(model, i):
        return model.x[i, i] == 0

    model.no_self = pyo.Constraint(model.N, rule=no_self_loop)

    def leave_once(model, i):
        return sum(model.x[i, j] for j in model.N if i != j) == 1

    model.leave = pyo.Constraint(model.N, rule=leave_once)

    def enter_once(model, j):
        return sum(model.x[i, j] for i in model.N if i != j) == 1

    model.enter = pyo.Constraint(model.N, rule=enter_once)

    # MTZ subtour elimination
    def mtz_constraint(model, i, j):
        if i == 0 or j == 0 or i == j:
            return pyo.Constraint.Skip
        return (
            model.u[i] - model.u[j] + len(cities) * model.x[i, j]
            <= len(cities) - 1
        )

    model.mtz = pyo.Constraint(model.N, model.N, rule=mtz_constraint)

    # Solve the model
    solver = pyo.SolverFactory("glpk")  # Use 'cbc' or 'gurobi' if installed
    solver.solve(model, tee=False)

    # Extract the tour
    tour = []
    current = 0
    visited = set([current])
    while True:
        for j in cities:
            if j != current and pyo.value(model.x[current, j]) > 0.5:
                tour.append((current, j))
                current = j
                visited.add(j)
                break
        if len(visited) == len(cities):
            tour.append((current, 0))  # return to start
            break

    best_tour_length = pyo.value(model.obj)
    return tour, best_tour_length


def example_1():
    import matplotlib.pyplot as plt
    import util

    coords = util.generate_tsp_coords()
    tour, length = solve_tsp_mip_mtz(coords)
    util.plot_2d_tour_coord(coords, tour)
    plt.show()


def example_2():
    import matplotlib.pyplot as plt
    import util

    bot = util.ur5e_dh()
    taskH = util.generate_random_dh_tasks(bot, 10)
    coords = util.make_coords_from_tasks_list(taskH)
    tour, length = solve_tsp_mip_mtz(coords)
    tour_simp = util.simplify_tour(tour)
    util.plot_tf_tour(taskH, names=None, tour=tour)
    plt.show()

    taskH = util.generate_linear_grid_tasks_transformation()
    coords = util.make_coords_from_tasks_list(taskH)
    tour, length = solve_tsp_mip_mtz(coords)
    tour_simp = util.simplify_tour(tour)
    util.plot_tf_tour(taskH, names=None, tour=tour)
    plt.show()

    # TOOK A VERY LONG TIME, UNSOLVABLE WITH MIP
    # taskH = generate_linear_dual_side_tasks_transformation()
    # coords = make_coords_from_tasks_list(taskH)
    # tour, length = solve_tsp_mip_mtz(coords)
    # tour_simp = simplify_tour(tour)
    # plot_tf_tour(taskH, names=None, tour=tour)
    # plt.show()


if __name__ == "__main__":
    example_1()
    example_2()
