import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)

def solve_tsp_aco(
    cities,
    num_ants=100,
    num_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    Q=100.0,
):
    num_cities = len(cities)

    # 1. Create Distance Matrix
    D = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                D[i, j] = np.linalg.norm(cities[i] - cities[j])
            else:
                D[i, j] = np.inf  # Prevent moving to the same city

    # # 1.1 Remove long distances (optional, based on your problem)
    # threshold = np.percentile(D[D != np.inf], 90)
    # D[D > threshold] = np.inf

    # 2. Initialize Pheromone Matrix with 1s
    pheromones = np.ones((num_cities, num_cities))

    # # 2.1 Ensure pheromones of long distances are set to 0 (optional)
    # pheromones[D == np.inf] = 0.001

    best_path = None
    best_distance = float("inf")

    # 3. Optimization Loop
    for _ in range(num_iterations):
        print(f"==>> Iteration {_ + 1}/{num_iterations}")
        all_paths = []
        all_distances = []

        for ant in range(num_ants):
            unvisited = list(range(num_cities))
            start_city = np.random.choice(unvisited)
            unvisited.remove(start_city)

            path = [start_city]
            current_city = start_city

            # Construct Tour
            while unvisited:
                # Compute transition probabilities
                raw_weights = []
                for next_city in unvisited:
                    pheromone_attr = pheromones[current_city, next_city] ** alpha
                    heuristic_attr = (1.0 / D[current_city, next_city]) ** beta
                    raw_weights.append(pheromone_attr * heuristic_attr)

                probabilities = np.array(raw_weights) / sum(raw_weights)

                # Biased random choice based on pheromones and visibility
                next_city = np.random.choice(unvisited, p=probabilities)
                unvisited.remove(next_city)
                path.append(next_city)
                current_city = next_city

            path.append(start_city)  # Complete the loop
            all_paths.append(path)

            # Calculate total tour distance
            total_dist = sum(D[path[i], path[i + 1]] for i in range(num_cities))
            all_distances.append(total_dist)

            # Keep track of global best
            if total_dist < best_distance:
                best_distance = total_dist
                best_path = path

        # 4. Pheromone Evaporation
        pheromones *= 1.0 - rho

        # 5. Pheromone Deposit (Symmetric update)
        for path, dist in zip(all_paths, all_distances):
            for i in range(num_cities):
                pheromones[path[i], path[i + 1]] += Q / dist
                pheromones[path[i + 1], path[i]] += Q / dist

    return best_path, best_distance


def solve_tsp_aco_vectorized(
    cities,
    num_ants=100,
    num_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    Q=100.0,
    seed=None,
):
    rng = np.random.default_rng(seed)

    cities = np.asarray(cities, dtype=np.float64)
    n = len(cities)

    # ------------------------------------------------------------
    # 1. Distance matrix
    # ------------------------------------------------------------
    diff = cities[:, None, :] - cities[None, :, :]
    D = np.linalg.norm(diff, axis=-1)

    # Prevent self-transition
    np.fill_diagonal(D, np.inf)

    # ------------------------------------------------------------
    # 2. Precompute heuristic
    #
    # visibility = 1 / distance
    # ------------------------------------------------------------
    eta = np.zeros_like(D)
    valid = np.isfinite(D) & (D > 0)
    eta[valid] = 1.0 / D[valid]

    # ------------------------------------------------------------
    # 3. Pheromone
    # ------------------------------------------------------------
    pheromone = np.ones((n, n), dtype=np.float64)

    best_path = None
    best_distance = np.inf

    ant_ids = np.arange(num_ants)

    # ------------------------------------------------------------
    # 4. Main ACO loop
    # ------------------------------------------------------------
    for iteration in range(num_iterations):

        # --------------------------------------------------------
        # All ants start at random cities
        # --------------------------------------------------------
        starts = rng.integers(0, n, size=num_ants)

        paths = np.empty((num_ants, n + 1), dtype=np.int32)
        paths[:, 0] = starts

        # visited[a, city]
        visited = np.zeros((num_ants, n), dtype=bool)
        visited[ant_ids, starts] = True

        current = starts.copy()

        # --------------------------------------------------------
        # Construct all tours simultaneously
        # --------------------------------------------------------
        for step in range(1, n):

            # pheromone[current] -> (num_ants, n)
            tau = pheromone[current]

            # eta[current] -> (num_ants, n)
            visibility = eta[current]

            # Transition weight
            weights = np.power(tau, alpha) * np.power(visibility, beta)

            # Already visited cities cannot be selected
            weights[visited] = 0.0

            # ----------------------------------------------------
            # Vectorized categorical sampling.
            #
            # argmax(log(weight) + Gumbel noise)
            # gives a sample proportional to weight.
            # ----------------------------------------------------
            log_weights = np.full_like(weights, -np.inf)

            valid_weights = weights > 0
            log_weights[valid_weights] = np.log(weights[valid_weights])

            gumbel = rng.gumbel(size=weights.shape)

            next_city = np.argmax(log_weights + gumbel, axis=1)

            # Store
            paths[:, step] = next_city

            # Update state
            visited[ant_ids, next_city] = True
            current = next_city

        # Return to starting city
        paths[:, n] = starts

        # --------------------------------------------------------
        # 5. Calculate ALL tour lengths at once
        # --------------------------------------------------------
        edge_from = paths[:, :-1]
        edge_to = paths[:, 1:]

        distances = D[edge_from, edge_to].sum(axis=1)

        # --------------------------------------------------------
        # 6. Global best
        # --------------------------------------------------------
        best_ant = np.argmin(distances)

        if distances[best_ant] < best_distance:
            best_distance = distances[best_ant]
            best_path = paths[best_ant].copy()

        # --------------------------------------------------------
        # 7. Evaporation
        # --------------------------------------------------------
        pheromone *= 1.0 - rho

        # --------------------------------------------------------
        # 8. Pheromone deposition
        #
        # Each ant contributes Q / distance to its edges.
        # --------------------------------------------------------
        deposits = Q / distances

        # Flatten all ant edges
        from_flat = edge_from.ravel()
        to_flat = edge_to.ravel()

        deposit_flat = np.repeat(deposits, n)

        # Symmetric pheromone update
        np.add.at(pheromone, (from_flat, to_flat), deposit_flat)
        np.add.at(pheromone, (to_flat, from_flat), deposit_flat)

        print(pheromone)
        # --------------------------------------------------------
        # Optional progress
        # --------------------------------------------------------
        print(
            f"Iteration {iteration + 1}/{num_iterations} "
            f"| best = {best_distance:.4f}"
        )

    return best_path, best_distance


# --- Execution ---
if __name__ == "__main__":
    # Define 5 structural city coordinates (X, Y)
    # city_coordinates = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [5, 5]])
    city_coordinates = np.random.rand(50, 2) * 100

    # best_route, shortest_dist = solve_tsp_aco(city_coordinates)
    best_route, shortest_dist = solve_tsp_aco_vectorized(city_coordinates)
    print(f"Optimal Path found: {best_route}")
    print(f"Shortest Distance:  {shortest_dist:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(city_coordinates[:, 0], city_coordinates[:, 1], color="blue")
    for i, city in enumerate(city_coordinates):
        ax.annotate(
            f"City {i}",
            (city[0], city[1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    for i in range(len(best_route) - 1):
        start_city = city_coordinates[best_route[i]]
        end_city = city_coordinates[best_route[i + 1]]
        ax.plot(
            [start_city[0], end_city[0]],
            [start_city[1], end_city[1]],
            color="red",
            linestyle="-",
        )
    ax.set_title("Optimal TSP Route using ACO")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.grid()
    plt.show()
