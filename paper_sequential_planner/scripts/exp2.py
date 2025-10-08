# RoboTSP + altconfig
from robotsp_solver import RoboTSPSolver
import numpy as np
import time
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
    # qinit = np.array([3.4, -1.22, 1.25, 0.0, 1.81, 0.0]) # altconfig better
    # qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0]) # altconfig better
    qinit = np.array([7.1, -1.22, 1.25, 0.0, 1.81, 0.0])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_(Htasks)
    rtspsolver.solve(Htasks, Hinit, qinit, numsolslist, Qlist)
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


def visualization():
    taskH = taskspace_pose()

    num_sols, ik_sols = solve_ik_(taskH)
    qik = ik_sols[11]
    print(qik)
    qselect = qik[1]
    print("qselect:", qselect)

    pose = []
    for i, H in enumerate(taskH):
        xyz, quat = util.tf_to_xyzquat(H)
        pose.append((xyz, quat))

    robot = UR5eBullet("gui")
    model_id = robot.load_models_other(Constants.model_list_shelf)

    for pi in pose:
        robot.draw_frame(pi[0], pi[1], 0.1, 2)

    q0 = [0.0, -1.1, 1.1, 0.0, 0.0, 0.0]
    robot.reset_array_joint_state(q0)
    # q1 = [0.0, -np.pi / 2, 0, 0, 0, 0]
    q1 = qselect
    path = np.linspace(q0, q1, 100)

    try:
        j = 0
        while True:
            q = path[j]
            p.stepSimulation()
            # robot.joint_viewer()
            time.sleep(1 / 240)
            if j < 99:
                robot.control_array_motors(q)
                j += 1

    except KeyboardInterrupt:
        robot.disconnect()


if __name__ == "__main__":
    # visualization()
    solve_robotsp_normal()
    solve_robotsp_altconfig()
