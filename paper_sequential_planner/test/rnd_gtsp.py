import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt


def generate_gtsp_coords():
    coords = {
        0: (10, 10),
        1: (12, 14),
        2: (14, 10),
        3: (50, 50),
        4: (52, 54),
        5: (54, 50),
        6: (90, 10),
        7: (92, 14),
        8: (94, 10),
    }
    clusters = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
    }
    return coords, clusters


def generate_gtsp_random_coords():
    num_clusters = 10
    cities_per_cluster = 3
    clusters = {}
    coords = {}
    city_id = 0

    for c in range(num_clusters):
        base_x = np.random.uniform(0, 100)
        base_y = np.random.uniform(0, 100)
        clusters[c] = []
        for _ in range(cities_per_cluster):
            x = base_x + np.random.uniform(-5, 5)
            y = base_y + np.random.uniform(-5, 5)
            coords[city_id] = (x, y)
            clusters[c].append(city_id)
            city_id += 1
    return coords, clusters


def plot_2d_cluster_tour_coord(coords, clusters, tour):
    """for gtsp"""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["r", "g", "b", "c", "m", "y"]

    # Plot all cities by cluster
    for c, city_ids in clusters.items():
        for i in city_ids:
            x, y = coords[i]
            ax.plot(x, y, "o", color=colors[c % len(colors)])
            ax.text(x + 0.2, y + 0.2, str(i), fontsize=12)

    for i, j in tour:
        xi, yi = coords[i]
        xj, yj = coords[j]
        ax.plot([xi, xj], [yi, yj], "k-")

    ax.set_title(f"Tour: {tour}")
    ax.axis("equal")


def solve_gtsp_mip_mtz(coords, clusters):
    all_cities = list(coords.keys())

    # Euclidean distances
    def euclidean(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        return np.linalg.norm([xi - xj, yi - yj])

    distances = {
        (i, j): euclidean(i, j) for i in all_cities for j in all_cities if i != j
    }

    # Pyomo Model
    model = pyo.ConcreteModel()
    model.N = pyo.Set(initialize=all_cities)
    model.x = pyo.Var(model.N, model.N, domain=pyo.Binary)  # tour edges
    model.y = pyo.Var(model.N, domain=pyo.Binary)  # selected cities
    model.u = pyo.Var(
        model.N, domain=pyo.NonNegativeReals, bounds=(0, len(clusters))
    )  # MTZ

    # Objective: minimize tour length
    def obj_rule(model):
        return sum(
            distances[i, j] * model.x[i, j]
            for i in model.N
            for j in model.N
            if i != j
        )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints
    # Visit exactly one city from each cluster
    def one_city_per_cluster(model, c):
        return sum(model.y[i] for i in clusters[c]) == 1

    model.one_per_cluster = pyo.Constraint(
        range(len(clusters)), rule=one_city_per_cluster
    )

    # Only allow edges between selected cities
    def only_selected_edges(model, i, j):
        if i == j:
            return model.x[i, j] == 0
        return model.x[i, j] <= model.y[i]

    model.valid_edge = pyo.Constraint(model.N, model.N, rule=only_selected_edges)

    # Enter and leave selected cities exactly once
    def in_out_flow(model, i):
        return sum(model.x[j, i] for j in model.N if j != i) == model.y[i]

    model.inflow = pyo.Constraint(model.N, rule=in_out_flow)

    def out_in_flow(model, i):
        return sum(model.x[i, j] for j in model.N if j != i) == model.y[i]

    model.outflow = pyo.Constraint(model.N, rule=out_in_flow)

    # MTZ constraints to eliminate subtours among selected cities
    def mtz_rule(model, i, j):
        if i == j or i == list(model.N)[0] or j == list(model.N)[0]:
            return pyo.Constraint.Skip
        return (
            model.u[i] - model.u[j] + len(clusters) * model.x[i, j]
            <= len(clusters) - 1
        )

    model.mtz = pyo.Constraint(model.N, model.N, rule=mtz_rule)

    # Solve
    solver = pyo.SolverFactory("glpk")
    print("Solving GTSP model...")
    solver.solve(model, tee=False)

    # Reconstruct ordered tour
    selected_cities = [i for i in all_cities if pyo.value(model.y[i]) > 0.5]
    edges = [
        (i, j)
        for i in selected_cities
        for j in selected_cities
        if i != j and pyo.value(model.x[i, j]) > 0.5
    ]
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    path = [edges[0][0]]
    current = edges[0][0]
    while len(path) <= len(edges):
        for nxt in adj[current]:
            if nxt not in path or (len(path) == len(edges) and nxt == path[0]):
                path.append(nxt)
                current = nxt
                break
    tour = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    best_tour_length = pyo.value(model.obj)
    return tour, best_tour_length


def example_1():
    import matplotlib.pyplot as plt

    # coords, clusters = generate_gtsp_random_coords()
    coords, clusters = generate_gtsp_coords()
    tour, best_tour_length = solve_gtsp_mip_mtz(coords, clusters)
    plot_2d_cluster_tour_coord(coords, clusters, tour)
    plt.show()


if __name__ == "__main__":
    example_1()
