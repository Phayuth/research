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

    stateStartColor = "black"
    stateStartFaceColor = "yellow"
    stateAppColor = "black"
    stateAppFaceColor = "green"
    stateGoalColor = "black"
    stateGoalFaceColor = "red"
    stateMarkersize = 7
    stateMarker = "o"

    pathColor = "blue"
    pathFaceColor = "plum"
    pathMarker = "o"
    pathMarkersize = 7


def fig_r2s_workspace_view():
    def fk_link(q):
        l1 = 2.0
        l2 = 2.0
        x0 = 0
        y0 = 0
        x1 = l1 * np.cos(q[0])
        y1 = l1 * np.sin(q[0])
        x2 = x1 + l2 * np.cos(q[0] + q[1])
        y2 = y1 + l2 * np.sin(q[0] + q[1])
        return np.array([[x0, y0], [x1, y1], [x2, y2]])

    rectangle = {
        # x, y, h, w
        "r0": [-0.7, 1.3, 2.0, 2.2],
        "r1": [2.0, -2.0, 1.0, 4.0],
        "r2": [-3.0, -3.0, 1.25, 2.0],
    }

    links_s = fk_link([-0.3, -np.pi / 2])
    links_g = fk_link([np.pi / 2, np.pi / 2])

    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    path = np.loadtxt(rsrc + "paper_r2s_snggoal_path.csv", delimiter=",")

    fig, ax = plt.subplots(1, 1, figsize=(1.7431, 1.7431))
    ax.axhline(y=0, color="black", linestyle="--", alpha=1, linewidth=1)
    ax.axvline(x=0, color="black", linestyle="--", alpha=1, linewidth=1)
    # obstacles
    for key in rectangle.keys():
        r = plt.Rectangle(
            (rectangle[key][0], rectangle[key][1]),
            rectangle[key][3],
            rectangle[key][2],
            color=PlotterConfig.obstColor,
        )
        ax.add_patch(r)

    # links_s
    ax.plot(
        links_s[:, 0],
        links_s[:, 1],
        color="black",
        linewidth=3,
        marker="o",
        markerfacecolor="yellow",
        markersize=5,
    )
    # links_g
    ax.plot(
        links_g[:, 0],
        links_g[:, 1],
        color="black",
        linewidth=3,
        marker="o",
        markerfacecolor="red",
        markersize=5,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim((-3, 3))
    ax.set_ylim((-3, 3))
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["-3", "-2", "-1", "0", "1", "2", "3"])
    ax.set_yticklabels(["-3", "-2", "-1", "0", "1", "2", "3"])
    fig.tight_layout(pad=0)
    plt.show()


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

    fig, ax = plt.subplots(1, 1, figsize=(2.4403, 2.4403))
    # tree
    # for u, v in graph.edges:
    #     u = graph.nodes[u]["coords"].rsplit(",")
    #     v = graph.nodes[v]["coords"].rsplit(",")
    #     ax.plot(
    #         [float(u[0]), float(v[0])],
    #         [float(u[1]), float(v[1])],
    #         color=PlotterConfig.treeColor,
    #         linewidth=PlotterConfig.globalLinewidth,
    #         marker=PlotterConfig.treeMarker,
    #         markerfacecolor=PlotterConfig.treeFaceColor,
    #         markersize=PlotterConfig.treeMarkersize,
    #         alpha=0.4,
    #     )

    # fake tree
    r = plt.Rectangle((-5, -7), 6, 14, color=PlotterConfig.treeColor, alpha=0.4)
    ax.add_patch(r)

    # collision
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )

    # path
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=PlotterConfig.pathColor,
        linewidth=PlotterConfig.globalLinewidth,
    )
    ax.plot(
        path2hack[:, 0],
        path2hack[:, 1],
        color=PlotterConfig.pathColor,
        linewidth=PlotterConfig.globalLinewidth,
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

    ax.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["$-2\\pi$", "$-\\pi$", "$\\theta_1$", "$\\pi$", "$2\\pi$"])
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["$-2\\pi$", "$-\\pi$", "$\\theta_2$", "$\\pi$", "$2\\pi$"])

    ax.set_xlim((-2 * np.pi, 2 * np.pi))
    ax.set_ylim((-2 * np.pi, 2 * np.pi))
    ax.axhline(y=0, color="green", linestyle=":", linewidth=3)
    ax.axvline(x=0, color="green", linestyle=":", linewidth=3)
    ax.axhline(y=2 * np.pi, color="red", linestyle="-", linewidth=3)
    ax.axhline(y=-2 * np.pi, color="red", linestyle="-", linewidth=3)
    ax.axvline(x=2 * np.pi, color="red", linestyle="-", linewidth=3)
    ax.axvline(x=-2 * np.pi, color="red", linestyle="-", linewidth=3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=0)
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
            alpha=0.4,
        )
        ax.plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color=PlotterConfig.treeColor,
            linewidth=PlotterConfig.globalLinewidth,
            marker=PlotterConfig.treeMarker,
            markerfacecolor=PlotterConfig.treeFaceColor,
            markersize=PlotterConfig.treeMarkersize,
            alpha=0.4,
        )

    # collision
    ax.plot(
        colp[:, 0],
        colp[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
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
        )
        ax.plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color=PlotterConfig.pathColor,
            linewidth=PlotterConfig.globalLinewidth,
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
        )
        ax.plot(
            [v[0], qvuw[0]],
            [v[1], qvuw[1]],
            color="red",
            linewidth=PlotterConfig.globalLinewidth,
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

    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((-np.pi, np.pi))
    ax.set_aspect("equal", adjustable="box")
    # disable axis for texture
    ax.axis("off")
    ax.axhline(y=0, color="red", linestyle="-", linewidth=3)
    ax.axvline(x=0, color="red", linestyle="-", linewidth=3)
    ax.axhline(y=0, color="green", linestyle=":", linewidth=3)
    ax.axvline(x=0, color="green", linestyle=":", linewidth=3)
    plt.show()


def fig_generate_labels():
    pass


if __name__ == "__main__":
    fig_r2s_workspace_view()
    # fig_r2s_snggoal_altgoals()
    # fig_so2s_snggoal_for_torus_texture()
    # fig_generate_labels()
