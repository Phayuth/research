import os
import numpy as np
from matplotlib import animation
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_planar_rr import RobotArm2DSimulator
from task_map import PaperICCAS2024

np.random.seed(9)
sim = RobotArm2DSimulator(PaperICCAS2024(), torusspace=True)


def fig_naive_extended_space():
    """
    planning in extended space for the sake of joint limit only.
    no alternative goals are consider.
    show complete cspace (obstacle, alternative goals, tree).
    show performance.
    show animation.
    """

    plannarConfigDualTreea = {
        "planner": 1,
        "eta": 0.3,
        "subEta": 0.05,
        "maxIteration": 1000,
        "simulator": sim,
        "nearGoalRadius": None,
        "rewireRadius": None,
        "endIterationID": 1,
        "printDebug": True,
        "localOptEnable": False,
    }

    # case 1
    # xStart = np.array([0.0, 0.0]).reshape(2,1)
    # xApp = np.array([3.0, 0.5]).reshape(2,1)
    # xApp = np.array([4.6, -5.77]).reshape(2,1)
    # xApp = np.array([4.6, 0.51318531]).reshape(2,1)
    # xApp = np.array([-1.68318531, -5.77]).reshape(2,1)
    # xApp = np.array([-1.68318531, 0.51318531]).reshape(2,1)

    # case 2
    xStart = np.array([-2, 2.5]).reshape(2, 1)
    xApp = np.array([3, -2]).reshape(2, 1)

    xGoal = xApp.copy()

    pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
    patha = pm.begin_planner()

    pm.plot_2d_complete()
    pm.plot_performance()
    sim.play_back_path(patha, animation)


def fig_torus_extended_space():
    """
    planning to alternative goal on extended torusspace.
    show complete cspace (obstacle, alternative goals, tree).
    show performance.
    show animation.
    """
    plannarConfig = {
        "planner": 9,
        "eta": 0.3,
        "subEta": 0.05,
        "maxIteration": 10000,
        "simulator": sim,
        "nearGoalRadius": None,
        "rewireRadius": None,
        "endIterationID": 1,
        "printDebug": True,
        "localOptEnable": False,
    }

    # xStart = np.array([0.0, 0.0]).reshape(2,1)
    # # xApp = np.array([4.6, -5.77]).reshape(2,1)
    # xApp = np.array([3.0, 0.5]).reshape(2,1)

    xStart = np.array([-2, 2.5]).reshape(2, 1)
    xApp = np.array([3, -2]).reshape(2, 1)

    xGoal = xApp.copy()

    pa = RRTPlannerAPI.init_alt_q_torus(xStart, xApp, xGoal, plannarConfig, None)
    patha = pa.begin_planner()

    pa.plot_2d_complete()
    pa.plot_performance()
    sim.play_back_path(patha, animation)
