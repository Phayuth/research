# RoboTSP + altconfig
from robotsp_solver import (
    RoboTSPSolver,
    collision_check_Qlist,
    remove_collision_Qlist,
    cspace_candidate_selection_dijkstra_multi_sink,
)
from robotsp_ompl_planner import OMPLPlanner
import numpy as np
import util
from plot_ur5e_bullet import UR5eBullet, Constants
import pybullet as p


def taskspace_pose():
    size = 4
    quat1 = [-0.707106, 0.0, 0.0, 0.707106]
    H11 = util.generate_linear_tasks_transformation(
        [-0.4, 0.6, 0.5], [0.4, 0.6, 0.5], quat1, size
    )
    H22 = util.generate_linear_tasks_transformation(
        [-0.4, 0.6, 0.2], [0.4, 0.6, 0.2], quat1, size
    )

    quat2 = [-0.5, -0.5, 0.5, 0.5]
    H33 = util.generate_linear_tasks_transformation(
        [-0.6, -0.4, 0.5], [-0.6, 0.4, 0.5], quat2, size
    )
    H44 = util.generate_linear_tasks_transformation(
        [-0.6, -0.4, 0.2], [-0.6, 0.4, 0.2], quat2, size
    )

    quat3 = [0.0, -0.707106, 0.707106, 0.0]
    H55 = util.generate_linear_tasks_transformation(
        [0.4, -0.6, 0.5], [-0.4, -0.6, 0.5], quat3, size
    )
    H66 = util.generate_linear_tasks_transformation(
        [0.4, -0.6, 0.2], [-0.4, -0.6, 0.2], quat3, size
    )
    return H11 + H22 + H33 + H44 + H55 + H66


def _taskspace_pose_in_dhframe(taskH):
    taskH_dh = []
    for H in taskH:
        H_dh = util.convert_urdf_to_dh_frame(H)
        taskH_dh.append(H_dh)
    return taskH_dh


def solve_ik_(taskH):
    bot = util.ur5e_dh()
    taskH_dh = _taskspace_pose_in_dhframe(taskH)
    num_sols, ik_sols = util.solve_ik_bulk(bot, taskH_dh)
    return num_sols, ik_sols


def solve_ik_altconfig_(taskH):
    bot = util.ur5e_dh()
    taskH_dh = _taskspace_pose_in_dhframe(taskH)
    num_sols, ik_sols = util.solve_ik_altconfig_bulk(bot, taskH_dh)
    return num_sols, ik_sols


def solve_fk_(q):
    bot = util.ur5e_dh()
    H = util.solve_fk(bot, q)
    H = util.convert_dh_to_urdf_frame(H)
    return H


def solve_robotsp_normal():
    print("===================== Solve RoboTSP normal =================")
    rtspsolver = RoboTSPSolver()

    Htasks = taskspace_pose()
    qinit = np.array([3.4, -1.22, 1.25, 0.0, 1.81, 0.0])  # altconfig better
    # qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0]) # altconfig better
    # qinit = np.array([7.1, -1.22, 1.25, 0.0, 1.81, 0.0])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_(Htasks)
    cf_tour, cf_costs = rtspsolver.solve(Htasks, Hinit, qinit, numsolslist, Qlist)
    rtspsolver.print_log()


def solve_robotsp_altconfig():
    print("===================== Solve RoboTSP altconfig =================")
    rtspsolver = RoboTSPSolver()

    Htasks = taskspace_pose()
    qinit = np.array([3.4, -1.22, 1.25, 0.0, 1.81, 0.0])  # altconfig better
    # qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0])  # altconfig better
    # qinit = np.array([7.1, -1.22, 1.25, 0.0, 1.81, 0.0])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_altconfig_(Htasks)
    cf_tour, cf_costs = rtspsolver.solve(Htasks, Hinit, qinit, numsolslist, Qlist)
    rtspsolver.print_log()


def solve_robotsp_grocery_picking_normal():
    robot = UR5eBullet("no_gui")
    model_id = robot.load_models_other(Constants.model_list_shelf)
    op = OMPLPlanner(robot.collision_check_at_config)

    def collision_check_individual(q):
        return robot.collision_check_at_config(q)

    def query_collisionfree_path(qa, qb):
        result = op.query_planning(qa, qb)
        if result is not None:
            path, cost = result
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

    rtspsolver = RoboTSPSolver(
        cspace_collisionfree_tour_func=cspace_collisionfree_tour,
        tspace_tsp_solver_method="exact_branch_and_bound",
    )

    Htasks = taskspace_pose()
    # qinit = np.array([3.4, -1.22, 1.25, 0.1, 1.81, 0.1])  # altconfig better
    qinit = np.array(
        [
            3.4 - 2 * np.pi,
            -1.22,
            1.25,
            0.1 - 2 * np.pi,
            1.81 - 2 * np.pi,
            0.1 - 2 * np.pi,
        ]
    )
    # qinit = np.array([7.1, -1.22, 1.25, 0.1, 1.81, 0.1])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_(Htasks)
    QlistCollision = collision_check_Qlist(Qlist, collision_check_individual)
    QlistFree, numsolsfree = remove_collision_Qlist(Qlist, QlistCollision)
    cf_tour, cf_costs = rtspsolver.solve(
        Htasks, Hinit, qinit, numsolsfree, QlistFree
    )
    rtspsolver.print_log()
    return cf_tour, cf_costs


def solve_robotsp_grocery_picking_altconfig():
    robot = UR5eBullet("no_gui")
    model_id = robot.load_models_other(Constants.model_list_shelf)
    op = OMPLPlanner(robot.collision_check_at_config)

    def collision_check_individual(q):
        return robot.collision_check_at_config(q)

    def query_collisionfree_path(qa, qb):
        result = op.query_planning(qa, qb)
        if result is not None:
            path, cost = result
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

    rtspsolver = RoboTSPSolver(
        cspace_collisionfree_tour_func=cspace_collisionfree_tour,
        tspace_tsp_solver_method="exact_branch_and_bound",
        cspace_candidate_selection_func=cspace_candidate_selection_dijkstra_multi_sink,
    )

    Htasks = taskspace_pose()
    # qinit = np.array([3.4, -1.22, 1.25, 0.1, 1.81, 0.1])  # altconfig better
    qinit = np.array(
        [
            3.4 - 2 * np.pi,
            -1.22,
            1.25,
            0.1 - 2 * np.pi,
            1.81 - 2 * np.pi,
            0.1 - 2 * np.pi,
        ]
    )
    # qinit = np.array([7.1, -1.22, 1.25, 0.1, 1.81, 0.1])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_(Htasks)
    QlistCollision = collision_check_Qlist(Qlist, collision_check_individual)
    QlistFree, _ = remove_collision_Qlist(Qlist, QlistCollision)
    numsolsfree, QlistFree = util.find_altconfig_bulk(QlistFree)
    cf_tour, cf_costs = rtspsolver.solve(
        Htasks, Hinit, qinit, numsolsfree, QlistFree
    )
    rtspsolver.print_log()
    return cf_tour, cf_costs


def visualization():
    # robot
    robot = UR5eBullet("gui")
    model_id = robot.load_models_other(Constants.model_list_shelf)
    robot.load_models_ghost(color=[0, 1, 0, 0.1])  # green ghost model

    taskH = taskspace_pose()
    pose = []
    for i, H in enumerate(taskH):
        xyz, quat = util.tf_to_xyzquat(H)
        pose.append((xyz, quat))
    for id, pi in enumerate(pose):
        robot.draw_frame(pi[0], pi[1], 0.1, 2, text=f"task{id}")

    tour = np.load("cf_tour_altconfig_2.npy", allow_pickle=True)
    path = retimer_tour(tour)

    robot.reset_array_joint_state(path[0])
    robot.reset_array_joint_state_ghost(path[-1], robot.ghost_model[0])
    try:
        j = 0
        while True:
            nkey = ord("n")
            bkey = ord("b")
            keys = p.getKeyboardEvents()
            if nkey in keys and keys[nkey] & p.KEY_WAS_TRIGGERED:
                q = path[j % path.shape[0]]
                robot.reset_array_joint_state(q)
                p.stepSimulation()
                j += 1
            elif bkey in keys and keys[bkey] & p.KEY_WAS_TRIGGERED:
                q = path[j % path.shape[0]]
                robot.reset_array_joint_state(q)
                p.stepSimulation()
                j -= 1
    except KeyboardInterrupt:
        robot.disconnect()

    # tour_normal = np.load("cf_tour_normal_2.npy", allow_pickle=True)
    # tour_altconfig = np.load("cf_tour_altconfig_2.npy", allow_pickle=True)

    # path_nm = retimer_tour(tour_normal)
    # path_ac = retimer_tour(tour_altconfig)

    # robot.reset_array_joint_state(path_nm[0])
    # robot.reset_array_joint_state_ghost(path_ac[0], robot.ghost_model[0])
    # try:
    #     j = 0
    #     while True:
    #         nkey = ord("n")
    #         bkey = ord("b")
    #         keys = p.getKeyboardEvents()
    #         if nkey in keys and keys[nkey] & p.KEY_WAS_TRIGGERED:
    #             q_nm = path_nm[j % path_nm.shape[0]]
    #             q_ac = path_ac[j % path_ac.shape[0]]
    #             robot.reset_array_joint_state(q_nm)
    #             robot.reset_array_joint_state_ghost(q_ac, robot.ghost_model[0])
    #             p.stepSimulation()
    #             j += 1
    #         elif bkey in keys and keys[bkey] & p.KEY_WAS_TRIGGERED:
    #             q_nm = path_nm[j % path_nm.shape[0]]
    #             q_ac = path_ac[j % path_ac.shape[0]]
    #             robot.reset_array_joint_state(q_nm)
    #             robot.reset_array_joint_state_ghost(q_ac, robot.ghost_model[0])
    #             p.stepSimulation()
    #             j -= 1
    # except KeyboardInterrupt:
    #     robot.disconnect()


def flatten_tour(cf_tour):
    t = []
    for pp in cf_tour:
        for pi in pp:
            t.append(pi)
    return np.array(t)


def retimer_tour(path_tour):
    _, first_occ_ind = np.unique(path_tour, axis=0, return_index=True)
    sorted_indices = np.sort(first_occ_ind)
    path_tour_nodup = path_tour[sorted_indices]

    path_tour_interp = []
    for i in range(path_tour_nodup.shape[0] - 1):
        pi_nm = np.linspace(path_tour_nodup[i], path_tour_nodup[i + 1], 10)
        for px in pi_nm:
            path_tour_interp.append(px)
    path_tour_interp = np.vstack(path_tour_interp)
    return path_tour_interp


if __name__ == "__main__":
    # visualization()
    # solve_robotsp_normal()
    # solve_robotsp_altconfig()

    # cf_tour_nm, cf_costs_nm = solve_robotsp_grocery_picking_normal()
    # cf_tour_nm = flatten_tour(cf_tour_nm)
    # np.save("cf_tour_normal_2.npy", cf_tour_nm)

    cf_tour_ac, cf_costs_ac = solve_robotsp_grocery_picking_altconfig()
    cf_tour_ac = flatten_tour(cf_tour_ac)
    np.save("cf_tour_altconfig_2.npy", cf_tour_ac)
