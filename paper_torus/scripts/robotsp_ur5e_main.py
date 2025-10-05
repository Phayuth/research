from termcolor import cprint
import numpy as np
import time
from util import (
    generate_random_dh_tasks,
    generate_random_task_transformation,
    generate_linear_grid_tasks_transformation,
    generate_linear_dual_side_tasks_transformation,
    generate_spiral_task_transformation,
    solve_ik,
    solve_fk,
    plot_tf,
    plot_tf_tour,
    ur5e_dh,
    generate_random_dh_tasks,
    solve_ik_altconfig,
    solve_ik_altconfig_bulk,
    solve_ik_bulk,
    pickle_load,
    rotate_tour_coords_format,
    rotate_tour_simplifiy_format,
    pickle_dump,
    make_ndarray_from_task_list,
    simplify_tour,
    expand_tour,
    make_coords_from_tasks_list,
    np_save,
    np_load,
    option_runner,
    path_configuration_length,
    nearest_neighbour_transformation,
)

from rnd_tsp import solve_tsp_mip_mtz
from rnd_tsp_nn import solve_tsp_nearest_neighbor
from robotsp_dijkstra_solver import robotsp_dijkstra_solve


bot = ur5e_dh()


def solve_taskspace_tsp_mip_mtz(taskH):
    """
    solve for sequence order in task space using
    travelling salesman problem (TSP) approach.
    """
    coords = make_coords_from_tasks_list(taskH)
    tour, length = solve_tsp_mip_mtz(coords)
    order = simplify_tour(tour)
    return tour, order


def solve_taskspace_tsp_nn(taskH):
    ndarray = make_ndarray_from_task_list(taskH)
    order, length = solve_tsp_nearest_neighbor(
        coords=ndarray, multi_start="all", use_two_opt=True
    )
    return None, order


def collision_check(q):
    return True


def collision_check_bulk(Q):
    colp = []
    for q in Q:
        c = collision_check(q)
        colp.append(c)
    return colp


def reorder_taskH(taskH, order):
    taskHreorder = []
    for o in order:
        taskHreorder.append(taskH[o])
    return taskHreorder


def configurations_in_order(taskHorder):
    numsols, Qorder = solve_ik_bulk(bot, taskHorder)
    return numsols, Qorder


def configurations_in_order_altconfig(taskHorder):
    numsols, Qorder = solve_ik_altconfig_bulk(bot, taskHorder)
    return numsols, Qorder


def solve_optimal_config_given_order(qinit, Qorder):
    """
    since we have a fixed order of tasks, this is simplified to dijkstra.
    it return
    optimal config = [(6,), (6,), (6,), ...]
    """
    optimal_config_tour = robotsp_dijkstra_solve(qinit, Qorder)
    return optimal_config_tour


def query_birrt(qa, qb):
    """
    query path from qa to qb using bidirectional RRT.
    """
    # fake bench time
    for _ in range(1000000):
        1 + 1
    fake_path = np.linspace(qa, qb, num=10)
    return fake_path


def solve_collision_free_tour(optimal_configs):
    collision_free_tour = []
    for i in range(len(optimal_configs) - 1):
        qa = optimal_configs[i]
        qb = optimal_configs[i + 1]
        path = query_birrt(qa, qb)
        collision_free_tour.append(path)
    return collision_free_tour


def solve_robotsp(taskH, qinit):
    # solving for order in taskspace
    t1 = time.time()
    hinit = solve_fk(bot, qinit)
    minid, minh = nearest_neighbour_transformation(taskH, hinit)
    # tour, order = solve_taskspace_tsp_mip_mtz(taskH)
    _, order = solve_taskspace_tsp_nn(taskH)
    order = rotate_tour_simplifiy_format(order, minid)
    tour = expand_tour(order)
    taskHreoder = reorder_taskH(taskH, order)
    t2 = time.time()

    # solving for configuration tree given the order (disregard collsion)
    t3 = time.time()
    _, config_in_order = configurations_in_order(taskHreoder)
    optimal_configs = solve_optimal_config_given_order(qinit, config_in_order)
    t4 = time.time()

    # solving for collision free path
    t5 = time.time()
    collision_free_tour = solve_collision_free_tour(optimal_configs)
    t6 = time.time()

    cprint(f"Time taken to solve TSP: {t2 - t1} sec", "green")
    cprint(f"Time taken to solve optimal config: {t4 - t3} sec", "green")
    cprint(f"Time taken to solve collision free tour: {t6 - t5} sec", "green")

    qpath = np.vstack(collision_free_tour)
    return tour, qpath


def solve_robotsp_altconfig(taskH, qinit):
    # solving for order in taskspace
    t1 = time.time()
    hinit = solve_fk(bot, qinit)
    minid, minh = nearest_neighbour_transformation(taskH, hinit)
    # tour, order = solve_taskspace_tsp_mip_mtz(taskH)
    _, order = solve_taskspace_tsp_nn(taskH)
    order = rotate_tour_simplifiy_format(order, minid)
    tour = expand_tour(order)
    taskHreoder = reorder_taskH(taskH, order)
    t2 = time.time()

    # solving for configuration tree given the order (disregard collsion)
    t3 = time.time()
    _, config_in_order = configurations_in_order_altconfig(taskHreoder)
    optimal_configs = solve_optimal_config_given_order(qinit, config_in_order)
    t4 = time.time()

    # solving for collision free path
    t5 = time.time()
    collision_free_tour = solve_collision_free_tour(optimal_configs)
    t6 = time.time()

    cprint(f"Time taken to solve TSP: {t2 - t1} sec", "green")
    cprint(f"Time taken to solve optimal config: {t4 - t3} sec", "green")
    cprint(f"Time taken to solve collision free tour: {t6 - t5} sec", "green")

    qpath = np.vstack(collision_free_tour)
    return tour, qpath


def example_solve_noaltconfig():
    import matplotlib.pyplot as plt

    # taskH = generate_linear_grid_tasks_transformation()
    # taskH = generate_linear_dual_side_tasks_transformation()
    taskH = generate_spiral_task_transformation()
    qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0])

    tour, qpath = solve_robotsp(taskH, qinit)
    qpathlength = path_configuration_length(qpath)
    cprint(f"Path length: {qpathlength}", "yellow")

    # saving log
    print("Path shape:", qpath.shape)
    np_save(qpath, "collision_free_tour.npy")
    plot_tf_tour(T=taskH, names=None, tour=tour)
    plt.show()


def example_solve_withaltconfig():
    import matplotlib.pyplot as plt

    # taskH = generate_linear_grid_tasks_transformation()
    # taskH = generate_linear_dual_side_tasks_transformation()
    taskH = generate_spiral_task_transformation()
    qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0])

    tour, qpath = solve_robotsp_altconfig(taskH, qinit)
    qpathlength = path_configuration_length(qpath)
    cprint(f"Path length: {qpathlength}", "yellow")

    # saving log
    print("Path shape:", qpath.shape)
    np_save(qpath, "collision_free_tour_altconfig.npy")
    plot_tf_tour(T=taskH, names=None, tour=tour)
    plt.show()


def plot_joint_time():
    from plot_joint_times import plot_joint_times, times_zero_to_one

    joint = np_load("collision_free_tour.npy")
    jointalt = np_load("collision_free_tour_altconfig.npy")

    times = times_zero_to_one(joint.shape[0])
    plot_joint_times(joint, times, jointalt)


if __name__ == "__main__":
    func = [
        example_solve_noaltconfig,
        example_solve_withaltconfig,
        plot_joint_time,
    ]
    option_runner(func, default_id="1")
