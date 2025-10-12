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


def fig_so2s_snggoal():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    graph = nx.read_graphml(rsrc + "paper_so2s_snggoal_planner_data.graphml")
    path = np.loadtxt(rsrc + "paper_so2s_snggoal_path.csv", delimiter=",")
    state = np.loadtxt(rsrc + "paper_so2s_snggoal_start_goal.csv", delimiter=",")
    colp = np.load(rsrc + "collisionpoint_so2s.npy")
    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

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
    plt.show()


def fig_so2s_snggoal_for_torus_texture():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    graph = nx.read_graphml(rsrc + "paper_so2s_snggoal_planner_data.graphml")
    path = np.loadtxt(rsrc + "paper_so2s_snggoal_path.csv", delimiter=",")
    state = np.loadtxt(rsrc + "paper_so2s_snggoal_start_goal.csv", delimiter=",")
    colp = np.load(rsrc + "collisionpoint_so2s.npy")
    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

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
    plt.show()


def fig_so2s_snggoal_path():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    path = np.loadtxt(rsrc + "paper_so2s_snggoal_path.csv", delimiter=",")
    times = np.linspace(0.0, 1.0, path.shape[0])

    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

    # plotting
    fig = plt.figure()
    gs = fig.add_gridspec(7, 1, height_ratios=[1, 1, 1, 0.3, 1, 1, 1], hspace=0)
    axes = [
        fig.add_subplot(gs[0, 0]),  # xtop
        fig.add_subplot(gs[1, 0]),  # x
        fig.add_subplot(gs[2, 0]),  # xbot
        fig.add_subplot(gs[4, 0]),  # ytop (skip gs[3,0] for gap)
        fig.add_subplot(gs[5, 0]),  # y
        fig.add_subplot(gs[6, 0]),  # ybot
    ]
    # Share x-axis across all subplots
    for i in range(1, len(axes)):
        axes[i].sharex(axes[0])

    # process data
    xpath = path.copy()
    xpath[:, 1] = times
    xpath = xpath[:, [1, 0]]
    ypath = path.copy()
    ypath[:, 0] = times

    # plot ----- TORUS ----
    for i in range(xpath.shape[0] - 1):
        u = xpath[i].reshape(2, 1)
        v = xpath[i + 1].reshape(2, 1)
        quvw = Utils.nearest_qb_to_qa(u, v, limt2, ignoreOrginal=False)
        qvuw = Utils.nearest_qb_to_qa(v, u, limt2, ignoreOrginal=False)
        axes[1].plot(
            [u[0], quvw[0]],
            [u[1], quvw[1]],
            color=PlotterConfig.pathColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.pathMarker,
            markerfacecolor=PlotterConfig.pathFaceColor,
            markersize=PlotterConfig.pathMarkersize,
        )
        axes[1].plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color=PlotterConfig.pathColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.pathMarker,
            markerfacecolor=PlotterConfig.pathFaceColor,
            markersize=PlotterConfig.pathMarkersize,
        )

    for i in range(ypath.shape[0] - 1):
        u = ypath[i].reshape(2, 1)
        v = ypath[i + 1].reshape(2, 1)
        quvw = Utils.nearest_qb_to_qa(u, v, limt2, ignoreOrginal=False)
        qvuw = Utils.nearest_qb_to_qa(v, u, limt2, ignoreOrginal=False)
        axes[4].plot(
            [u[0], quvw[0]],
            [u[1], quvw[1]],
            color=PlotterConfig.pathColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.pathMarker,
            markerfacecolor=PlotterConfig.pathFaceColor,
            markersize=PlotterConfig.pathMarkersize,
        )
        axes[4].plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color=PlotterConfig.pathColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.pathMarker,
            markerfacecolor=PlotterConfig.pathFaceColor,
            markersize=PlotterConfig.pathMarkersize,
        )

    # plot normal ----------------
    # proess unwrap
    pathunwrap = Utils.unwrap_so2_path(path.T)
    xpathunwrap = pathunwrap[0, :]
    ypathunwrap = pathunwrap[1, :]
    print(xpathunwrap)
    axes[1].plot(
        times,
        xpathunwrap,
        color="gray",
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )
    axes[2].plot(
        times,
        xpathunwrap,
        color="gray",
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )
    axes[4].plot(
        times,
        ypathunwrap,
        color="gray",
        linewidth=PlotterConfig.globalLinewidth,
        marker=PlotterConfig.pathMarker,
        markerfacecolor=PlotterConfig.pathFaceColor,
        markersize=PlotterConfig.pathMarkersize,
    )

    # set limits and ticks
    for ax in axes:
        ax.set_xlim((0.0, 1.0))
        ax.grid(True)
    # xtop
    axes[0].set_ylim((np.pi, 2 * np.pi))
    axes[0].tick_params(left=False, labelleft=False)
    axes[0].tick_params(bottom=False, labelbottom=False)
    # x
    axes[1].set_ylim((-np.pi, np.pi))
    axes[1].tick_params(bottom=False, labelbottom=False)
    axes[1].set_yticklabels(["$-\\pi\$", "0", "$\\pi\$"])
    axes[1].set_yticks([-3.1, 0, 3.1])
    # xbot
    axes[2].set_ylim((-2 * np.pi, -np.pi))
    axes[2].tick_params(left=False, labelleft=False)

    # ytop
    axes[3].set_ylim((np.pi, 2 * np.pi))
    axes[3].tick_params(left=False, labelleft=False)
    axes[3].tick_params(bottom=False, labelbottom=False)
    # y
    axes[4].set_ylim((-np.pi, np.pi))
    axes[4].tick_params(bottom=False, labelbottom=False)
    axes[4].set_yticklabels(["$-\\pi\$", "0", "$\\pi\$"])
    axes[4].set_yticks([-3.1, 0, 3.1])
    # ybot
    axes[5].set_ylim((-2 * np.pi, -np.pi))
    axes[5].tick_params(left=False, labelleft=False)

    plt.show()


def fig_so2s_snggoal_path_2():
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    path = np.loadtxt(rsrc + "paper_so2s_snggoal_path.csv", delimiter=",")
    times = np.linspace(0.0, 1.0, path.shape[0])

    limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

    # plotting
    fig, axes = plt.subplots(2, 1, sharex=True)

    # process data
    xpath = path.copy()
    xpath[:, 1] = times
    xpath = xpath[:, [1, 0]]
    ypath = path.copy()
    ypath[:, 0] = times

    pathunwrap = Utils.unwrap_so2_path(path.T)
    xpathunwrap = pathunwrap[0, :]
    ypathunwrap = pathunwrap[1, :]

    # plot normal ----------------
    axes[0].plot(
        times,
        xpathunwrap,
        color="gray",
        linewidth=PlotterConfig.globalLinewidth + 5,
        linestyle="--",
    )
    axes[1].plot(
        times,
        ypathunwrap,
        color="gray",
        linewidth=PlotterConfig.globalLinewidth + 5,
        linestyle="--",
    )

    # plot ----- TORUS ----
    for i in range(xpath.shape[0] - 1):
        u = xpath[i].reshape(2, 1)
        v = xpath[i + 1].reshape(2, 1)
        quvw = Utils.nearest_qb_to_qa(u, v, limt2, ignoreOrginal=False)
        qvuw = Utils.nearest_qb_to_qa(v, u, limt2, ignoreOrginal=False)
        axes[0].plot(
            [u[0], quvw[0]],
            [u[1], quvw[1]],
            linewidth=PlotterConfig.globalLinewidth,
            color="blue",
        )
        axes[0].plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            linewidth=PlotterConfig.globalLinewidth,
            color="blue",
        )

    for i in range(ypath.shape[0] - 1):
        u = ypath[i].reshape(2, 1)
        v = ypath[i + 1].reshape(2, 1)
        quvw = Utils.nearest_qb_to_qa(u, v, limt2, ignoreOrginal=False)
        qvuw = Utils.nearest_qb_to_qa(v, u, limt2, ignoreOrginal=False)
        axes[1].plot(
            [u[0], quvw[0]],
            [u[1], quvw[1]],
            linewidth=PlotterConfig.globalLinewidth,
            color="red",
        )
        axes[1].plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            linewidth=PlotterConfig.globalLinewidth,
            color="red",
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(np.pi, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(-np.pi, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(np.pi, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(-np.pi, color="black", linestyle="--", linewidth=1)

    axes[0].set_yticklabels(["$-2\\pi$", "$-\\pi$", "0", "$\\pi$", "$2\\pi$"])
    axes[0].set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    axes[1].set_yticklabels(["$-2\\pi$", "$-\\pi$", "0", "$\\pi$", "$2\\pi$"])
    axes[1].set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])

    axes[1].set_xlim((0.0, 1.0))
    axes[0].set_ylim((-2 * np.pi, 2 * np.pi))
    axes[1].set_ylim((-2 * np.pi, 2 * np.pi))

    plt.show()


if __name__ == "__main__":
    fig_so2s_snggoal()
    # fig_so2s_snggoal_for_torus_texture()
    # fig_so2s_snggoal_path()
    # fig_so2s_snggoal_path_2()
