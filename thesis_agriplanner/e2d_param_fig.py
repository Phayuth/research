import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt

from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
from task_map import Thesis2024
from simulator.sim_planar_rr import RobotArm2DSimulator
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI

rc = 0.8
cx = -3.0
cy = 0.5
g1 = rbt.conv_polar_to_cartesian(rc, 0.0, cx, cy)
g2 = rbt.conv_polar_to_cartesian(rc, 0.5, cx, cy)  # intended collision
g3 = rbt.conv_polar_to_cartesian(rc, -0.5, cx, cy)

sim = RobotArm2DSimulator(Thesis2024(), torusspace=False)

robot = sim.robot
q0 = np.array([0, 0]).reshape(2, 1)
q1 = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=1)
q2 = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=1)
q3 = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=1)

thetas = np.hstack((q0, q1, q2, q3))


def fig_wspace_candidates():
    """
    generate figure in the workspace before planning consists of
    - workspace obstacles
    - workspace initial posture
    - workspace goals posture (6 postures with elbow up and elbow down)
    """

    fig, ax = plt.subplots()
    sim.plot_taskspace()
    robot.plot_arm(thetas, ax, shadow=False)
    circle1 = plt.Circle((cx, cy), rc, color="r")
    ax.add_patch(circle1)
    plt.show()


def fig_wspace_motion():
    """
    generate figure motion animation in the workspace after planning
    """
    plannarConfigDualTreea = {
        "planner": 5,
        "eta": 0.3,
        "subEta": 0.05,
        "maxIteration": 2000,
        "simulator": sim,
        "nearGoalRadius": None,
        "rewireRadius": None,
        "endIterationID": 1,
        "printDebug": True,
        "localOptEnable": True,
    }

    # single planner
    xStart = np.array([0, 0]).reshape(2, 1)
    xApp = q1
    xGoal = q1

    # multiple goal planner
    # xStart = np.array([0, 0]).reshape(2, 1)
    # xApp = [q1, q2, q3]
    # xGoal = [q1, q2, q3]

    pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
    patha = pm.begin_planner()
    pm.plot_performance()

    fig, ax = plt.subplots()
    ax.set_xlim(sim.taskspace.xlim)
    ax.set_ylim(sim.taskspace.ylim)
    sim.plot_taskspace()
    robot.plot_arm(patha, ax, shadow=False)
    circle1 = plt.Circle((cx, cy), rc, color="r")
    ax.add_patch(circle1)
    plt.show()
