import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from spatial_geometry.utils import Utils
from simulator.sim_planar_rrr import RobotArm2DRRRSimulator
from task_map import PaperIJCAS2025

np.random.seed(9)
plt.rcParams["svg.fonttype"] = "none"


class PaperLengths:
    latex_linewidth_inch = 3.19423
    latex_columnwidth_inch = 3.19423
    latex_textwidth_inch = 6.68459
    mmtopoint = 72 / 25.4
    inchtopoint = 72


rc = 0.8
cx = -3.0
cy = 0.5
g1 = np.array([cx, cy, 0.0 - np.pi]).reshape(3, 1)
g2 = np.array([cx, cy, 0.5 - np.pi]).reshape(3, 1)  # intended collision
g3 = np.array([cx, cy, -0.5 - np.pi]).reshape(3, 1)

sim = RobotArm2DRRRSimulator(PaperIJCAS2025())

robot = sim.robot
q0 = np.array([0, 0, 0]).reshape(3, 1)

q1 = robot.inverse_kinematic_geometry(g1, elbow_option=1)
q2 = robot.inverse_kinematic_geometry(g2, elbow_option=1)
q3 = robot.inverse_kinematic_geometry(g3, elbow_option=1)

q1i = robot.inverse_kinematic_geometry(g1, elbow_option=0)
q2i = robot.inverse_kinematic_geometry(g2, elbow_option=0)
q3i = robot.inverse_kinematic_geometry(g3, elbow_option=0)

q1i = Utils.wrap_to_pi(q1i)
q2i = Utils.wrap_to_pi(q2i)
q3i = Utils.wrap_to_pi(q3i)

xStart = q0
xApp = [q1, q3, q1i, q2i, q3i]
xGoal = [q1, q3, q1i, q2i, q3i]

thetas = np.hstack((q0, q1, q2, q3, q1i, q2i, q3i))


def fig_wspace_candidates():
    """
    generate figure in the workspace before planning consists of
    - workspace obstacles
    - workspace initial postures
    - workspace goal postures
    """
    # figure setup
    wd = 3.19423 * 0.8
    ht = 3.19423 * 0.8
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")
    ax.set_xlim(sim.taskspace.xlim)
    ax.set_ylim(sim.taskspace.ylim)
    ax.axhline(color="gray", alpha=0.4)
    ax.axvline(color="gray", alpha=0.4)

    # remove stuff
    ax.grid(False)
    ax.set_xticklabels(["$x$"])
    ax.set_yticklabels(["$y$"])
    ax.set_xticks([0])
    ax.set_yticks([0])

    sim.plot_taskspace()
    robot.plot_arm(
        thetas,
        ax,
        shadow=False,
        colors=["indigo", "red", "blue", "green", "red", "blue", "green"],
    )
    c = plt.Circle(
        (cx, cy),
        rc,
        alpha=0.4,
        edgecolor="g",
        facecolor=None,
        fill=False,
        linestyle="--",
    )
    ax.add_patch(c)
    ax.plot([cx], [cy], "g*")
    # plt.show()
    plt.savefig("~/exp3_wspace.png", bbox_inches="tight", transparent=True)


def fig_wspace_motion():
    """
    generate figure motion animation in the workspace after planning
    """

    planconfig = {
        "planner": 13,
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

    pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, planconfig)
    path = pm.begin_planner()

    # figure setup
    wd = 3.19423 * 0.8
    ht = 3.19423 * 0.8
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")
    ax.set_xlim(sim.taskspace.xlim)
    ax.set_ylim(sim.taskspace.ylim)
    ax.axhline(color="gray", alpha=0.4)
    ax.axvline(color="gray", alpha=0.4)

    # remove stuff
    ax.grid(False)
    ax.set_xticklabels(["$x$"])
    ax.set_yticklabels(["$y$"])
    ax.set_xticks([0])
    ax.set_yticks([0])

    sim.plot_taskspace()
    robot.plot_arm(path, ax, shadow=True)
    c = plt.Circle(
        (cx, cy),
        rc,
        alpha=0.4,
        edgecolor="g",
        facecolor=None,
        fill=False,
        linestyle="--",
    )
    ax.add_patch(c)
    ax.plot([cx], [cy], "g*")

    plt.savefig("~/exp3_motion.png", bbox_inches="tight", transparent=True)
