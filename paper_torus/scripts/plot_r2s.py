import os
import matplotlib.pyplot as plt
from simulator.sim_planar_rr import RobotArm2DSimulator
import numpy as np
from rnd_task_map import PaperTorusIFAC2025
import networkx as nx
from spatial_geometry.utils import Utils


class PlotterConfig:
    globalLinewidth = 1

    obstColor = "darkcyan"
    obstFaceColor = "darkcyan"
    obstMarker = "o"
    obstMarkersize = 1.5

    treeColor = "darkgray"
    treeFaceColor = None
    treeMarker = None
    treeMarkersize = None

    stateStartColor = "blue"
    stateStartFaceColor = "yellow"
    stateAppColor = "blue"
    stateAppFaceColor = "green"
    stateGoalColor = "blue"
    stateGoalFaceColor = "red"
    stateMarkersize = 7
    stateMarker = "o"

    pathColor = "blue"
    pathFaceColor = "plum"
    pathMarker = "o"
    pathMarkersize = 7


def generate_cspace_point():
    """
    generate cspace figure with so2(minimal space), extended space.
    switch between torus=true/false for the corresponding
    """
    torus = True
    env = RobotArm2DSimulator(PaperTorusIFAC2025(), torusspace=torus)
    colp = env.generate_collision_points()
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    if torus:
        np.save(rsrc + "collisionpoint_exts.npy", colp)
    else:
        np.save(rsrc + "collisionpoint_so2s.npy", colp)


def view_cspace_point():
    """
    view the generated cspace points.
    """
    torus = True
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"

    if torus:
        colp = np.load(rsrc + "collisionpoint_exts.npy")
    else:
        colp = np.load(rsrc + "collisionpoint_so2s.npy")

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )

    if torus:
        ax.set_xlim([-2 * np.pi, 2 * np.pi])
        ax.set_ylim([-2 * np.pi, 2 * np.pi])
        ax.axhline(y=np.pi, color="gray", alpha=0.4)
        ax.axhline(y=-np.pi, color="gray", alpha=0.4)
        ax.axvline(x=np.pi, color="gray", alpha=0.4)
        ax.axvline(x=-np.pi, color="gray", alpha=0.4)
    else:
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
    plt.show()


def view_cspace_presentation_so2():
    """
    view the generated cspace points.
    """
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    colp = np.load(rsrc + "collisionpoint_so2s.npy")

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])

    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    q1 = np.array([2.5, 0.0]).reshape(2, 1)
    q2 = np.array([-2.0, -2.5]).reshape(2, 1)
    quvw = Utils.nearest_qb_to_qa(q1, q2, limt2, ignoreOrginal=False)
    qvuw = Utils.nearest_qb_to_qa(q2, q1, limt2, ignoreOrginal=False)
    ax.plot(
        [q1[0], quvw[0]],
        [q1[1], quvw[1]],
        color="red",
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )
    ax.plot(
        [q2[0], qvuw[0]],
        [q2[1], qvuw[1]],
        color="red",
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )
    plt.show()


def view_cspace_presentation_extended():
    """
    view the generated cspace points.
    """
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    colp = np.load(rsrc + "collisionpoint_exts.npy")

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )
    ax.set_xlim([-2 * np.pi, 2 * np.pi])
    ax.set_ylim([-2 * np.pi, 2 * np.pi])
    ax.axhline(y=np.pi, color="gray", alpha=0.4)
    ax.axhline(y=-np.pi, color="gray", alpha=0.4)
    ax.axvline(x=np.pi, color="gray", alpha=0.4)
    ax.axvline(x=-np.pi, color="gray", alpha=0.4)

    q1 = np.array([2.5, 0.0]).reshape(2, 1)
    q2 = np.array([4.0, -2.5]).reshape(2, 1)
    ax.plot(
        [q1[0], q2[0]],
        [q1[1], q2[1]],
        color="red",
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )

    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    qalt = Utils.find_alt_config(q2, limt2, filterOriginalq=False)
    ax.plot(
        qalt[0, :],
        qalt[1, :],
        color="red",
        linewidth=0,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )

    plt.show()


def view_cspace_presentation_altconfig():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    colp = np.load(rsrc + "collisionpoint_exts.npy")

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )
    ax.set_xlim([-2 * np.pi, 2 * np.pi])
    ax.set_ylim([-2 * np.pi, 2 * np.pi])
    ax.axhline(y=np.pi, color="gray", alpha=0.4)
    ax.axhline(y=-np.pi, color="gray", alpha=0.4)
    ax.axvline(x=np.pi, color="gray", alpha=0.4)
    ax.axvline(x=-np.pi, color="gray", alpha=0.4)

    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    q = np.array([-2.0, -2.5]).reshape(2, 1)
    qalt = Utils.find_alt_config(q, limt2, filterOriginalq=False)
    ax.plot(
        qalt[0, :],
        qalt[1, :],
        color="red",
        linewidth=0,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )

    plt.show()


def view_cspace_presentation_extended_joint_violation():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    colp = np.load(rsrc + "collisionpoint_exts.npy")

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )
    ax.set_xlim([-2 * np.pi, 2 * np.pi])
    ax.set_ylim([-2 * np.pi, 2 * np.pi])
    ax.axhline(y=0, color="gray", alpha=0.4)
    ax.axhline(y=np.pi, color="gray", alpha=0.4)
    ax.axhline(y=-np.pi, color="gray", alpha=0.4)
    ax.axhline(y=2 * np.pi, color="gray", alpha=0.4)
    ax.axhline(y=-2 * np.pi, color="gray", alpha=0.4)

    ax.axvline(x=0, color="gray", alpha=0.4)
    ax.axvline(x=np.pi, color="gray", alpha=0.4)
    ax.axvline(x=-np.pi, color="gray", alpha=0.4)
    ax.axvline(x=2 * np.pi, color="gray", alpha=0.4)
    ax.axvline(x=-2 * np.pi, color="gray", alpha=0.4)

    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    q1 = np.array([-2.0, -2.5]).reshape(2, 1)
    q2 = np.array([0.5, -2.5]).reshape(2, 1)
    q1alt = Utils.find_alt_config(q1, limt2, filterOriginalq=False)
    q2alt = Utils.find_alt_config(q2, limt2, filterOriginalq=False)
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)
    print(f"q1alt: {q1alt}")
    print(f"q2alt: {q2alt}")

    ax.plot(
        [q1[0], q2[0]],
        [q1[1], q2[1]],
        color="blue",
    )
    ax.plot(
        [4.283, 0.5 + 2 * np.pi],
        [-2.5, -2.5],
        color="blue",
    )
    ax.plot(
        [q1[0]],
        [q1[1]],
        color="yellow",
        linewidth=0,
        marker=PlotterConfig.stateMarker,
        markerfacecolor=PlotterConfig.stateStartFaceColor,
        markersize=PlotterConfig.stateMarkersize,
    )
    ax.plot(
        [q2[0]],
        [q2[1]],
        color="red",
        linewidth=0,
        marker=PlotterConfig.stateMarker,
        markerfacecolor="red",
        markersize=PlotterConfig.stateMarkersize,
    )
    ax.plot(
        [4.283],
        [-2.5],
        color="yellow",
        linewidth=0,
        marker=PlotterConfig.stateMarker,
        markerfacecolor="yellow",
        markersize=PlotterConfig.stateMarkersize,
    )

    ax.plot(
        [-np.pi, np.pi, np.pi, -np.pi, -np.pi],
        [-np.pi, -np.pi, np.pi, np.pi, -np.pi],
        color="green",
        linewidth=3,
        linestyle="--",
    )
    # ax.plot(
    #     q1alt[0, :],
    #     q1alt[1, :],
    #     color="yellow",
    #     linewidth=0,
    #     marker=PlotterConfig.pathMarker,
    #     markerfacecolor="yellow",
    #     markersize=PlotterConfig.pathMarkersize,
    # )
    # ax.plot(
    #     q2alt[0, :],
    #     q2alt[1, :],
    #     color="red",
    #     linewidth=0,
    #     marker=PlotterConfig.pathMarker,
    #     markerfacecolor="red",
    #     markersize=PlotterConfig.pathMarkersize,
    # )

    plt.show()


def fig_r2s_cartgoal():
    """
    generate and visualize figure with minimal and extended space.
    """
    torus = True
    env = RobotArm2DSimulator(PaperTorusIFAC2025(), torusspace=torus)
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    thetas = np.loadtxt(rsrc + "paper_r2s_cartgoal_path.csv", delimiter=",")
    print(thetas.shape)
    env.plot_view(thetas.T)

    fig, ax = plt.subplots(1, 1)
    colp = np.load(rsrc + "collisionpoint_exts.npy")
    ax.plot(colp[:, 0], colp[:, 1], "o", markersize=1, color="gray", alpha=0.5)
    ax.plot(thetas[:, 0], thetas[:, 1], "o-", markersize=2, color="red", alpha=0.5)
    if torus:
        ax.set_xlim([-2 * np.pi, 2 * np.pi])
        ax.set_ylim([-2 * np.pi, 2 * np.pi])
        ax.axhline(y=np.pi, color="gray", alpha=0.4)
        ax.axhline(y=-np.pi, color="gray", alpha=0.4)
        ax.axvline(x=np.pi, color="gray", alpha=0.4)
        ax.axvline(x=-np.pi, color="gray", alpha=0.4)
    else:
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
    plt.show()


def fig_r2s_altgoal():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    graph = nx.read_graphml(rsrc + "paper_r2s_altgoal_planner_data.graphml")
    path = np.loadtxt(rsrc + "paper_r2s_altgoal_path.csv", delimiter=",")
    state = np.loadtxt(
        rsrc + "paper_r2s_altgoal_start_goal.csv",
        delimiter=",",
    )
    colp = np.load(rsrc + "collisionpoint_exts.npy")

    # task space
    torus = True
    env = RobotArm2DSimulator(PaperTorusIFAC2025(), torusspace=torus)
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    env.plot_view(path.T)

    fig, ax = plt.subplots(1, 1)
    # tree
    for u, v in graph.edges:
        u = graph.nodes[u]["coords"].rsplit(",")
        v = graph.nodes[v]["coords"].rsplit(",")
        ax.plot(
            [float(u[0]), float(v[0])],
            [float(u[1]), float(v[1])],
            color=PlotterConfig.treeColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.treeMarker,
            markerfacecolor=PlotterConfig.treeFaceColor,
            markersize=PlotterConfig.treeMarkersize,
        )

    # path
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=PlotterConfig.pathColor,
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )

    # state start
    ax.plot(
        state[0, 0],
        state[0, 1],
        color=PlotterConfig.stateStartColor,
        linewidth=0,
        marker=PlotterConfig.stateMarker,
        markerfacecolor=PlotterConfig.stateStartFaceColor,
        markersize=PlotterConfig.stateMarkersize,
    )
    # state goals
    ax.plot(
        state[1:, 0],
        state[1:, 1],
        color=PlotterConfig.stateGoalColor,
        linewidth=0,
        marker=PlotterConfig.stateMarker,
        markerfacecolor=PlotterConfig.stateGoalFaceColor,
        markersize=PlotterConfig.stateMarkersize,
    )

    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )

    ax.set_xlim((-2 * np.pi, 2 * np.pi))
    ax.set_ylim((-2 * np.pi, 2 * np.pi))
    ax.axhline(y=np.pi, color="gray", alpha=0.4)
    ax.axhline(y=-np.pi, color="gray", alpha=0.4)
    ax.axvline(x=np.pi, color="gray", alpha=0.4)
    ax.axvline(x=-np.pi, color="gray", alpha=0.4)
    plt.show()


def fig_r2s_snggoal():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    graph = nx.read_graphml(rsrc + "paper_r2s_snggoal_planner_data.graphml")
    path = np.loadtxt(rsrc + "paper_r2s_snggoal_path.csv", delimiter=",")
    state = np.loadtxt(rsrc + "paper_r2s_snggoal_start_goal.csv", delimiter=",")
    colp = np.load(rsrc + "collisionpoint_exts.npy")

    # task space
    torus = True
    env = RobotArm2DSimulator(PaperTorusIFAC2025(), torusspace=torus)
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    env.plot_view(path.T)

    fig, ax = plt.subplots(1, 1)
    # tree
    for u, v in graph.edges:
        u = graph.nodes[u]["coords"].rsplit(",")
        v = graph.nodes[v]["coords"].rsplit(",")
        ax.plot(
            [float(u[0]), float(v[0])],
            [float(u[1]), float(v[1])],
            color=PlotterConfig.treeColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.treeMarker,
            markerfacecolor=PlotterConfig.treeFaceColor,
            markersize=PlotterConfig.treeMarkersize,
        )

    # path
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=PlotterConfig.pathColor,
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )

    # state start
    ax.plot(
        state[0, 0],
        state[0, 1],
        color=PlotterConfig.stateStartColor,
        linewidth=0,
        marker=PlotterConfig.stateMarker,
        markerfacecolor=PlotterConfig.stateStartFaceColor,
        markersize=PlotterConfig.stateMarkersize,
    )
    # state goals
    ax.plot(
        state[1:, 0],
        state[1:, 1],
        color=PlotterConfig.stateGoalColor,
        linewidth=0,
        marker=PlotterConfig.stateMarker,
        markerfacecolor=PlotterConfig.stateGoalFaceColor,
        markersize=PlotterConfig.stateMarkersize,
    )
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )

    plt.show()


def fig_r2s_prm():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    graph = nx.read_graphml(rsrc + "saved_planner.graphml")
    colp = np.load(rsrc + "collisionpoint_exts.npy")

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    # tree
    for u, v in graph.edges:
        u = graph.nodes[u]["coords"].rsplit(",")
        v = graph.nodes[v]["coords"].rsplit(",")
        ax.plot(
            [float(u[0]), float(v[0])],
            [float(u[1]), float(v[1])],
            color=PlotterConfig.treeColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.treeMarker,
            markerfacecolor=PlotterConfig.treeFaceColor,
            markersize=PlotterConfig.treeMarkersize,
        )

    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )

    plt.show()


if __name__ == "__main__":
    while True:
        func = [
            generate_cspace_point,
            view_cspace_point,
            view_cspace_presentation_so2,
            view_cspace_presentation_extended,
            view_cspace_presentation_altconfig,
            view_cspace_presentation_extended_joint_violation,
            fig_r2s_cartgoal,
            fig_r2s_altgoal,
            fig_r2s_snggoal,
            fig_r2s_prm,
        ]

        for i, f in enumerate(func, start=1):
            print(f"{i}: {f.__name__}")

        arg = input("Enter argument number (` to exit): ")

        if arg == "`":
            print("Exiting...")
            break
        elif arg.isdigit() and 1 <= int(arg) <= len(func):
            func[int(arg) - 1]()
