import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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


def fig_r2s_snggoal_altgoals():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    graph = nx.read_graphml(rsrc + "paper_r2s_altgoal_planner_data.graphml")
    path = np.loadtxt(rsrc + "paper_r2s_altgoal_path.csv", delimiter=",")
    path2 = np.loadtxt(rsrc + "paper_r2s_snggoal_path.csv", delimiter=",")
    state = np.loadtxt(rsrc + "paper_r2s_altgoal_start_goal.csv", delimiter=",")
    colp = np.load(rsrc + "collisionpoint_exts.npy")

    # hack path
    path2hack = np.empty((5, 2))
    mid = np.linspace(path2[0, :], path2[1, :], 4)
    path2hack[0, :] = path2[0, :]
    path2hack[[1, 2], :] = mid[1:3, :]
    path2hack[3, :] = path2[1, :]
    path2hack[4, :] = path2[2, :]

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
    ax.plot(
        path2hack[:, 0],
        path2hack[:, 1],
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
    ax.axhline(y=0, color="green", alpha=0.4, linestyle=":", linewidth=5)
    ax.axvline(x=0, color="green", alpha=0.4, linestyle=":", linewidth=5)
    ax.axhline(y=2 * np.pi, color="red", alpha=0.4, linestyle="--", linewidth=5)
    ax.axhline(y=-2 * np.pi, color="red", alpha=0.4, linestyle="--", linewidth=5)
    ax.axvline(x=2 * np.pi, color="red", alpha=0.4, linestyle="--", linewidth=5)
    ax.axvline(x=-2 * np.pi, color="red", alpha=0.4, linestyle="--", linewidth=5)
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def fig_so2s_snggoal_for_torus_texture():
    """
    plan to multiple goals and wrap down to pi for torus texture
    """
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    graph = nx.read_graphml(rsrc + "paper_so2s_snggoal_planner_data.graphml")
    # graph = nx.read_graphml(rsrc + "paper_r2s_snggoal_planner_data.graphml")
    path = np.loadtxt(rsrc + "paper_so2s_snggoal_path.csv", delimiter=",")
    path2 = np.loadtxt(rsrc + "paper_r2s_snggoal_path.csv", delimiter=",")
    state = np.loadtxt(rsrc + "paper_so2s_snggoal_start_goal.csv", delimiter=",")
    colp = np.load(rsrc + "collisionpoint_so2s.npy")
    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

    path2hack = np.empty((5, 2))
    mid = np.linspace(path2[0, :], path2[1, :], 4)
    path2hack[0, :] = path2[0, :]
    path2hack[[1, 2], :] = mid[1:3, :]
    path2hack[3, :] = path2[1, :]
    path2hack[4, :] = path2[2, :]

    # wrap to pi
    path2 = path2hack.copy()
    for i in range(path2.shape[0]):
        for j in range(path2.shape[1]):
            path2[i, j] = Utils.wrap_to_pi(path2[i, j])

    # plotting
    fig, ax = plt.subplots(1, 1)
    # tree
    for u, v in graph.edges:
        u = graph.nodes[u]["coords"].rsplit(",")
        u = np.array(u).astype(np.float32).reshape(2, 1)
        v = graph.nodes[v]["coords"].rsplit(",")
        v = np.array(v).astype(np.float32).reshape(2, 1)
        quvw = Utils.nearest_qb_to_qa(u, v, limt2, ignoreOrginal=False)
        qvuw = Utils.nearest_qb_to_qa(v, u, limt2, ignoreOrginal=False)
        ax.plot(
            [u[0], quvw[0]],
            [u[1], quvw[1]],
            color=PlotterConfig.treeColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.treeMarker,
            markerfacecolor=PlotterConfig.treeFaceColor,
            markersize=PlotterConfig.treeMarkersize,
        )
        ax.plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color=PlotterConfig.treeColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.treeMarker,
            markerfacecolor=PlotterConfig.treeFaceColor,
            markersize=PlotterConfig.treeMarkersize,
        )

    # path
    for i in range(path.shape[0] - 1):
        u = path[i].reshape(2, 1)
        v = path[i + 1].reshape(2, 1)
        quvw = Utils.nearest_qb_to_qa(u, v, limt2, ignoreOrginal=False)
        qvuw = Utils.nearest_qb_to_qa(v, u, limt2, ignoreOrginal=False)
        ax.plot(
            [u[0], quvw[0]],
            [u[1], quvw[1]],
            color=PlotterConfig.pathColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.pathMarker,
            markerfacecolor=PlotterConfig.pathFaceColor,
            markersize=PlotterConfig.pathMarkersize,
        )
        ax.plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color=PlotterConfig.pathColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.pathMarker,
            markerfacecolor=PlotterConfig.pathFaceColor,
            markersize=PlotterConfig.pathMarkersize,
        )
    # path2
    for i in range(path2.shape[0] - 1):
        u = path2[i].reshape(2, 1)
        v = path2[i + 1].reshape(2, 1)
        quvw = Utils.nearest_qb_to_qa(u, v, limt2, ignoreOrginal=False)
        qvuw = Utils.nearest_qb_to_qa(v, u, limt2, ignoreOrginal=False)
        ax.plot(
            [u[0], quvw[0]],
            [u[1], quvw[1]],
            color="red",
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.pathMarker,
            markerfacecolor=PlotterConfig.pathFaceColor,
            markersize=PlotterConfig.pathMarkersize,
        )
        ax.plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color="red",
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

    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((-np.pi, np.pi))
    ax.set_aspect("equal", adjustable="box")
    # disable axis for texture
    ax.axis("off")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=5)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=5)
    ax.axhline(y=0, color="green", linestyle=":", linewidth=3)
    ax.axvline(x=0, color="green", linestyle=":", linewidth=3)
    plt.show()


if __name__ == "__main__":
    fig_r2s_snggoal_altgoals()
    fig_so2s_snggoal_for_torus_texture()
