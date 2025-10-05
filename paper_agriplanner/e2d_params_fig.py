import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np
from simulator.sim_planar_rr import RobotArm2DSimulator
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
from spatial_geometry.utils import Utils
from task_map import PaperIJCAS2025
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI

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
g1 = rbt.conv_polar_to_cartesian(rc, 0.0, cx, cy)
g2 = rbt.conv_polar_to_cartesian(rc, 0.5, cx, cy)  # intended collision
g3 = rbt.conv_polar_to_cartesian(rc, -0.5, cx, cy)

sim = RobotArm2DSimulator(PaperIJCAS2025())

robot = sim.robot
q0 = np.array([0, 0]).reshape(2, 1)

q1 = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=1)
q2 = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=1)
q3 = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=1)

q1i = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=0)
q2i = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=0)
q3i = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=0)

q1i = Utils.wrap_to_pi(q1i)
q2i = Utils.wrap_to_pi(q2i)
q3i = Utils.wrap_to_pi(q3i)


xStart = q0
xApp = [q1, q2, q3, q1i, q2i, q3i]
xGoal = [q1, q2, q3, q1i, q2i, q3i]

thetas = np.hstack((q0, q1, q2, q3, q1i, q2i, q3i))


def naive_cspace_plot():
    """
    naively plot space by use all the point. making the plot so slow.
    """
    wd = 6.68459 * 0.32
    ht = 6.68459 * 0.32
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")
    sim.plot_cspace(ax)
    # ax.set_xlabel("$\\theta_1$")
    # ax.set_ylabel("$\\theta_2$")
    ax.set_xticklabels(["$-\\pi$", "$\\theta_1$", "$\\pi$"])
    ax.set_yticklabels(["$-\\pi$", "$\\theta_2$", "$\\pi$"])
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.tick_params(axis="both", which="major", direction="in", length=5, width=1, colors="black", grid_color="gray", grid_alpha=0.2)
    # ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.1)
    plt.savefig("~/exp1_cspace.png", bbox_inches="tight")


def better_cspace_plot():
    """
    better plot space by cluster nearby point into a group and use concave hull to find shape
    """
    from sklearn.cluster import DBSCAN
    from e2d_concavehull import ConcaveHull

    xy_points = np.load("./datasave/planner_ijcas_data/collisionpoint_so2s.npy")

    num_clusters = 5
    kmeans = DBSCAN()
    cluster_labels = kmeans.fit_predict(xy_points)
    cluster_lists = [[] for _ in range(num_clusters)]
    for i, label in enumerate(cluster_labels):
        cluster_lists[label].append(xy_points[i])

    wd = 0.30 * PaperLengths.latex_textwidth_inch
    ht = 0.30 * PaperLengths.latex_textwidth_inch
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")

    # colors = ["red", "green", "blue", "orange", "purple"]
    colors = ["#2ca08980"] * num_clusters
    for i, cluster in enumerate(cluster_lists):
        cluster = np.array(cluster)

        # convex patch
        # hull = ConvexHull(cluster)
        # boundary_points = cluster[hull.vertices]
        # cluster_polygon = plt.Polygon(boundary_points, edgecolor=colors[i], facecolor="none")
        # ax.add_patch(cluster_polygon)
        # plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f"Cluster {i + 1}")

        # concave patch
        hull = ConcaveHull(cluster, 5)
        boundary_points = hull.calculate()
        # cluster_polygon = plt.Polygon(boundary_points, edgecolor=colors[i], facecolor=colors[i], alpha=0.5)
        cluster_polygon = plt.Polygon(
            boundary_points,
            edgecolor="k",
            facecolor=colors[i],
            linewidth=0.2 * PaperLengths.mmtopoint,
        )
        ax.add_patch(cluster_polygon)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    ax.set_xticklabels(["$-\pi\$", "$\\theta_1\$", "$\pi\$"])
    ax.set_yticklabels(["$-\pi\$", "$\\theta_2\$", "$\pi\$"])
    ax.set_xticks([-2.8, 0, 2.8])
    ax.set_yticks([-2.8, 0, 2.8])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.2,
        labelsize=0.1,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.1,
        labelsize=0.1,
    )
    # plt.savefig("~/exp1_cspace.png", bbox_inches="tight")
    plt.savefig(
        "~/exp1_cspace.svg", bbox_inches="tight", format="svg", transparent=True
    )


def plot_cobs_concavehull(ax):
    """
    better plot space by cluster nearby point into a group and use concave hull to find shape
    """
    from sklearn.cluster import DBSCAN
    from e2d_concavehull import ConcaveHull

    xy_points = np.load("./datasave/planner_ijcas_data/collisionpoint_so2s.npy")

    num_clusters = 5
    kmeans = DBSCAN()
    cluster_labels = kmeans.fit_predict(xy_points)
    cluster_lists = [[] for _ in range(num_clusters)]
    for i, label in enumerate(cluster_labels):
        cluster_lists[label].append(xy_points[i])

    colors = ["#2ca08980"] * num_clusters
    for i, cluster in enumerate(cluster_lists):
        cluster = np.array(cluster)

        # concave patch
        hull = ConcaveHull(cluster, 5)
        boundary_points = hull.calculate()
        # cluster_polygon = plt.Polygon(boundary_points, edgecolor=colors[i], facecolor=colors[i], alpha=0.5)
        cluster_polygon = plt.Polygon(
            boundary_points,
            edgecolor="k",
            facecolor=colors[i],
            linewidth=0.2 * PaperLengths.mmtopoint,
        )
        ax.add_patch(cluster_polygon)


def fig_cpsace_state():
    """
    generate figure for cspace state consists of
    - cspace obstacles
    - cspace initial state
    - cspace goals state
    - no tree
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

    wd = 0.30 * PaperLengths.latex_textwidth_inch
    ht = 0.30 * PaperLengths.latex_textwidth_inch
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")

    plot_cobs_concavehull(ax)
    pm.plot_2d_tree_path_state_external(ax)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    ax.set_xticklabels(["$-\pi\$", "$\\theta_1\$", "$\pi\$"])
    ax.set_yticklabels(["$-\pi\$", "$\\theta_2\$", "$\pi\$"])
    ax.set_xticks([-2.8, 0, 2.8])
    ax.set_yticks([-2.8, 0, 2.8])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.2,
        labelsize=0.1,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.1,
        labelsize=0.1,
    )

    # plt.savefig("~/exp1_cspacestate.png", bbox_inches="tight", transparent=True)
    plt.savefig(
        "~/exp1_cspacestate.svg",
        bbox_inches="tight",
        format="svg",
        transparent=True,
    )


def fig_cspace_complete():
    """
    generate figure for cspace complete consists of
    - cspace obstacles
    - cspace initial state
    - cspace goals state
    - tree

    instruction:
    - change the planner id
        - 16, Birrt
        - 12, RRTConnect
        - 13, RRTStarConnect
    - change the planner localOptEnable
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

    wd = 0.30 * PaperLengths.latex_textwidth_inch
    ht = 0.30 * PaperLengths.latex_textwidth_inch
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")

    plot_cobs_concavehull(ax)
    pm.plot_2d_tree_path_state_external(ax)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    ax.set_xticklabels(["$-\pi\$", "$\\theta_1\$", "$\pi\$"])
    ax.set_yticklabels(["$-\pi\$", "$\\theta_2\$", "$\pi\$"])
    ax.set_xticks([-2.8, 0, 2.8])
    ax.set_yticks([-2.8, 0, 2.8])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.2,
        labelsize=0.1,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.1,
        labelsize=0.1,
    )

    # exp1_birrt
    # exp1_birrt_lg
    if planconfig["planner"] == 16:
        str_p = "exp1_birrt"
    elif planconfig["planner"] == 12:
        str_p = "exp1_rrtcnt"
    elif planconfig["planner"] == 13:
        str_p = "exp1_rrtstarcnt"

    if planconfig["localOptEnable"]:
        str_p += "_lg"

    plt.savefig(
        f"~/{str_p}.svg", bbox_inches="tight", format="svg", transparent=True
    )


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

    # ax.grid(False)
    # ax.set_xticks([-4, -2, 0, 2, 4])
    # ax.set_yticks([-4, -2, 0, 2, 4])
    # ax.set_xlabel("$x$", fontsize=11)
    # ax.set_ylabel("$y$", fontsize=11)

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

    plt.savefig("~/exp1_motion.png", bbox_inches="tight", transparent=True)


def fig_wspace_candidates():
    """
    generate figure in the workspace before planning consists of
    - workspace obstacles
    - workspace initial posture
    - workspace goals posture (6 postures with elbow up and elbow down)
    """
    # figure setup
    # wd = 3.19423 * 0.8
    # ht = 3.19423 * 0.8
    wd = 0.30 * PaperLengths.latex_textwidth_inch
    ht = 0.30 * PaperLengths.latex_textwidth_inch
    fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect("equal")

    ax.set_xlim(sim.taskspace.xlim)
    ax.set_ylim(sim.taskspace.ylim)
    ax.axhline(color="gray", alpha=0.4)
    ax.axvline(color="gray", alpha=0.4)
    ax.grid(False)
    ax.set_xticklabels(["$x\$"])
    ax.set_yticklabels(["$y\$"])
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.2,
        labelsize=0.1,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=0,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.1,
        labelsize=0.1,
    )

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

    # plt.savefig("~/exp1_wspace.png", bbox_inches="tight", transparent=True)
    plt.savefig(
        "~/exp1_wspace.svg", bbox_inches="tight", format="svg", transparent=True
    )
