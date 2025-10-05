import numpy as np


def build_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """
    coords: shape (n,2) array of points
    returns: (n,n) distance matrix
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def tour_length(tour: np.ndarray, D: np.ndarray) -> float:
    return D[tour, np.roll(tour, -1)].sum()


def nearest_neighbor(D: np.ndarray, start: int = 0) -> np.ndarray:
    n = D.shape[0]
    unvisited = np.ones(n, dtype=bool)
    tour = np.empty(n, dtype=int)
    tour[0] = start
    unvisited[start] = False
    curr = start
    for i in range(1, n):
        dists = np.where(unvisited, D[curr], np.inf)
        nxt = np.argmin(dists)
        tour[i] = nxt
        unvisited[nxt] = False
        curr = nxt
    return tour


def two_opt(tour: np.ndarray, D: np.ndarray) -> np.ndarray:
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for k in range(i + 2, n if i > 0 else n - 1):
                c, d = tour[k], tour[(k + 1) % n]
                # gain if we swap edges
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < -1e-12:
                    tour[i + 1 : k + 1] = tour[i + 1 : k + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return tour


def solve_tsp_nearest_neighbor(
    coords=None,
    dist_matrix=None,
    multi_start="all",
    use_two_opt=True,
    seed=0,
):
    """
    coords: np.ndarray of shape (n,2) OR dist_matrix: (n,n)
    multi_start: 'all' | int (number of random starts)
    """
    if coords is not None:
        D = build_dist_matrix(np.asarray(coords))
    else:
        D = np.asarray(dist_matrix)
    n = D.shape[0]

    if multi_start == "all":
        starts = np.arange(n)
    elif isinstance(multi_start, int):
        rng = np.random.default_rng(seed)
        starts = rng.choice(n, size=min(multi_start, n), replace=False)
    else:
        starts = np.array([multi_start])

    best_tour, best_tour_length = None, np.inf
    for s in starts:
        tour = nearest_neighbor(D, s)
        if use_two_opt:
            tour = two_opt(tour, D)
        L = tour_length(tour, D)
        if L < best_tour_length:
            best_tour, best_tour_length = tour.copy(), L

    return best_tour, best_tour_length


def inspect_tour_wspace_and_cspace():
    import matplotlib.pyplot as plt
    from robot.nonmobile.planar_rr import PlanarRR

    np.random.seed(99)
    robot = PlanarRR()

    # xy = np.array(
    #     [
    #         [-1, -1],
    #         [-1, 1],
    #         [1, 1],
    #         [1, -1],
    #     ]
    # )

    # wspaceorder = [0, 2, 1, 3]
    # xyordered = xy[wspaceorder, :]

    # wspace compute order
    xy = np.random.uniform(-1, 1, size=(10, 2))
    wspaceorder, l = solve_tsp_nearest_neighbor(
        coords=xy, multi_start="all", use_two_opt=True
    )
    xyordered = xy[wspaceorder, :]

    # cspace
    qiksol = []
    for pt in xy:
        qsolu = robot.inverse_kinematic_geometry(pt.reshape(2, 1), elbow_option=0)
        qsold = robot.inverse_kinematic_geometry(pt.reshape(2, 1), elbow_option=1)
        qiksol.append(qsolu.T)
        # qiksol.append(qsold)
    qiksol = np.vstack(qiksol)
    qiksol_ordered = qiksol[wspaceorder, :]

    cspaceorder, l = solve_tsp_nearest_neighbor(
        coords=qiksol, multi_start="all", use_two_opt=True
    )
    cordered = qiksol[cspaceorder, :]

    xycordered = xy[cspaceorder, :]

    # plot
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(
        xyordered[:, 0], xyordered[:, 1], "ro-", label="tour solved at w"
    )
    for i in range(len(xy)):
        ax[0, 0].annotate(f"P{i}", (xy[i, 0], xy[i, 1]))
    ax[0, 0].grid()
    ax[0, 0].set_aspect("equal", adjustable="box")
    ax[0, 0].set_title(f"W order{wspaceorder}")
    ax[0, 0].legend()

    #
    ax[0, 1].plot(
        xycordered[:, 0], xycordered[:, 1], "go--", label="tour solved at c"
    )
    for i in range(len(xy)):
        ax[0, 1].annotate(f"P{i}", (xy[i, 0], xy[i, 1]))
    ax[0, 1].grid()
    ax[0, 1].set_aspect("equal", adjustable="box")
    ax[0, 1].set_title(f"C order{cspaceorder}")
    ax[0, 1].legend()

    # 1 axis
    ax[1, 0].plot(
        qiksol_ordered[:, 0], qiksol_ordered[:, 1], "ro-", label="tour solved at w"
    )
    for i in range(len(qiksol_ordered)):
        ax[1, 0].annotate(f"Q{i}", (qiksol[i, 0], qiksol[i, 1]))
    ax[1, 0].grid()
    ax[1, 0].set_aspect("equal", adjustable="box")
    ax[1, 0].set_title(f"W order{wspaceorder}")
    ax[1, 0].set_xlim(-4, 4)
    ax[1, 0].set_ylim(-4, 4)
    ax[1, 0].legend()

    # 1 axis
    ax[1, 1].plot(cordered[:, 0], cordered[:, 1], "go--", label="tour solved at c")
    for i in range(len(qiksol_ordered)):
        ax[1, 1].annotate(f"Q{i}", (qiksol[i, 0], qiksol[i, 1]))
    ax[1, 1].grid()
    ax[1, 1].set_aspect("equal", adjustable="box")
    ax[1, 1].set_title(f"C order{cspaceorder}")
    ax[1, 1].set_xlim(-4, 4)
    ax[1, 1].set_ylim(-4, 4)
    ax[1, 1].legend()

    plt.show()


def example_1():
    import time
    from termcolor import cprint
    import matplotlib.pyplot as plt
    from util import (
        plot_2d_tour,
        generate_ndarray,
    )

    stime = time.time()
    coords = generate_ndarray()
    tour, length = solve_tsp_nearest_neighbor(
        coords=coords, multi_start="all", use_two_opt=True
    )
    etime = time.time()
    cprint(f"Time taken: {etime - stime:.4f} seconds", "green")
    cprint(f"Tour: {tour}", "blue")
    cprint(f"Length: {length}", "blue")
    plot_2d_tour(coords, tour)
    plt.show()


def example_2():
    import matplotlib.pyplot as plt
    from util import (
        expand_tour,
        generate_linear_tasks_transformation,
        generate_linear_dual_side_tasks_transformation,
        generate_linear_grid_tasks_transformation,
        generate_spiral_task_transformation,
        make_ndarray_from_task_list,
        plot_tf_tour,
    )

    # taskH = generate_linear_tasks_transformation()
    # taskH = generate_linear_dual_side_tasks_transformation()
    # taskH = generate_linear_grid_tasks_transformation()
    taskH = generate_spiral_task_transformation()
    coords = make_ndarray_from_task_list(taskH)
    tour, length = solve_tsp_nearest_neighbor(
        coords=coords, multi_start="all", use_two_opt=True
    )
    print("Tour:", tour)
    print("Length:", length)
    tourexap = expand_tour(tour)
    plot_tf_tour(taskH, names=None, tour=tourexap)
    plt.show()


if __name__ == "__main__":
    inspect_tour_wspace_and_cspace()
    # example_1()
    # example_2()
