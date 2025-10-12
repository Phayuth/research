# RoboTSP + altconfig
from robotsp_solver import (
    RoboTSPSolver,
    collision_check_Qlist,
    remove_collision_Qlist,
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
    # qinit = np.array([3.4, -1.22, 1.25, 0.0, 1.81, 0.0]) # altconfig better
    # qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0])  # altconfig better
    qinit = np.array([7.1, -1.22, 1.25, 0.0, 1.81, 0.0])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_altconfig_(Htasks)
    rtspsolver.solve(Htasks, Hinit, qinit, numsolslist, Qlist)
    rtspsolver.print_log()


def solve_robotsp_normal_collision():
    robot = UR5eBullet("no_gui")
    model_id = robot.load_models_other(Constants.model_list_shelf)
    op = OMPLPlanner(robot.collision_check_at_config)

    def collision_check_individual(q):
        return robot.collision_check_at_config(q)

    def query_collisionfree_path(qa, qb):
        path = op.query_planning(qa, qb)
        cost = 0.0
        for i in range(len(path)):
            cost += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
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
        cspace_collisionfree_tour_func=cspace_collisionfree_tour
    )

    Htasks = taskspace_pose()
    qinit = np.array([3.4, -1.22, 1.25, 0.0, 1.81, 0.0])  # altconfig better
    # qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0]) # altconfig better
    # qinit = np.array([7.1, -1.22, 1.25, 0.0, 1.81, 0.0])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_(Htasks)
    QlistCollision = collision_check_Qlist(Qlist, collision_check_individual)
    QlistFree = remove_collision_Qlist(Qlist, QlistCollision)
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

    taskH = taskspace_pose()
    pose = []
    for i, H in enumerate(taskH):
        xyz, quat = util.tf_to_xyzquat(H)
        pose.append((xyz, quat))
    for id, pi in enumerate(pose):
        robot.draw_frame(pi[0], pi[1], 0.1, 2, text=f"task{id}")

    tour = np.load("cf_tour.npy", allow_pickle=True)
    path = []
    for i in range(tour.shape[0] - 1):
        pi = np.linspace(tour[i], tour[i + 1], 10)
        for px in pi:
            path.append(px)
    path = np.vstack(path)
    # path = tour

    robot.reset_array_joint_state(path[0])
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


if __name__ == "__main__":
    # visualization()
    # solve_robotsp_normal()
    # solve_robotsp_altconfig()

    cf_tour, cf_costs = solve_robotsp_normal_collision()
    print(cf_tour)
    t = []
    for pp in cf_tour:
        for pi in pp:
            t.append(pi)
    print("CF", t)

    # cf_tour = np.array(t)
    # np.save("cf_tour.npy", cf_tour)
    # print(cf_tour)
