import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from planner.sampling_based.rrt_component import Node, RRTComponent
import time


class TwoDOF:

    def __init__(self) -> None:
        self.configLimit = [[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]]
        self.configDoF = len(self.configLimit)
        self.xs = np.array([0.0, 0.0]).reshape(2, 1)
        self.xg = np.array([1.0, 1.0]).reshape(2, 1)


class ThreeDOF:

    def __init__(self) -> None:
        self.configLimit = [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
        self.configDoF = len(self.configLimit)
        self.xs = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        self.xg = np.array([1.0, 1.0, 1.0]).reshape(3, 1)


class FourDOF:

    def __init__(self) -> None:
        self.configLimit = [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
        self.configDoF = len(self.configLimit)
        self.xs = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)
        self.xg = np.array([1.0, 1.0, 1.0, 1.0]).reshape(4, 1)


class FiveDOF:

    def __init__(self) -> None:
        self.configLimit = [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
        self.configDoF = len(self.configLimit)
        self.xs = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(5, 1)
        self.xg = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).reshape(5, 1)


class SixDOF:

    def __init__(self) -> None:
        self.configLimit = [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
        self.configDoF = len(self.configLimit)
        self.xs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1)
        self.xg = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(6, 1)


class GlobalSampling(RRTComponent):

    def __init__(self, xStart=None, xGoal=None, config=None) -> None:
        super().__init__(config)

        self.timehist = []
        starttime = time.perf_counter_ns()
        for i in range(10000):
            self.uni_sampling()
            self.timehist.append(time.perf_counter_ns())
        self.elaptime = np.array(self.timehist) - np.array(starttime)
        self.avgtime = np.mean(self.elaptime)
        self.mintime = np.min(self.elaptime)
        self.maxtime = np.max(self.elaptime)
        self.stdtime = np.std(self.elaptime)
        # self.totaltime = (endtime - starttime) / 10000


class InformedSampling(RRTComponent):

    def __init__(self, xStart=None, xGoal=None, config=None) -> None:
        super().__init__(config)
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)

        # informed sampling properties
        self.cBestNow = 5.5
        self.C = self.rotation_to_world(
            self.xStart, self.xGoal
        )  # hyperellipsoid rotation axis
        self.cMin = self.distance_between_config(self.xStart, self.xGoal)
        self.xCenter = (self.xStart.config + self.xGoal.config) / 2

        self.timehist = []
        starttime = time.perf_counter_ns()
        for i in range(10000):
            self.informed_sampling(self.xCenter, self.cBestNow, self.cMin, self.C)
            self.timehist.append(time.perf_counter_ns())
        self.elaptime = np.array(self.timehist) - np.array(starttime)
        self.avgtime = np.mean(self.elaptime)
        self.mintime = np.min(self.elaptime)
        self.maxtime = np.max(self.elaptime)
        self.stdtime = np.std(self.elaptime)

        # self.totaltime = (endtime - starttime) / 10000


class LocalGapSampling(RRTComponent):

    def __init__(self, xStart=None, xGoal=None, config=None) -> None:
        super().__init__(config)
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)

        self.anchorPath = [self.xStart, self.xGoal]
        self.localPath = [self.xStart, self.xGoal]
        self.numSegSamplingNode = len(self.localPath) - 1

        self.timehist = []
        starttime = time.perf_counter_ns()
        for i in range(10000):
            self.local_path_sampling(
                self.anchorPath, self.localPath, self.numSegSamplingNode
            )
            self.timehist.append(time.perf_counter_ns())
        self.elaptime = np.array(self.timehist) - np.array(starttime)
        self.avgtime = np.mean(self.elaptime)
        self.mintime = np.min(self.elaptime)
        self.maxtime = np.max(self.elaptime)
        self.stdtime = np.std(self.elaptime)

        # self.totaltime = (endtime - starttime) / 10000


if __name__ == "__main__":
    dim = [2, 3, 4, 5, 6]
    globals = []
    globalsmin = []
    globalsmax = []
    globalsstd = []
    infoms = []
    infomsmin = []
    infomsmax = []
    infomsstd = []
    locals = []
    localsmin = []
    localsmax = []
    localsstd = []

    sim = [TwoDOF, ThreeDOF, FourDOF, FiveDOF, SixDOF]

    for i in range(len(sim)):
        ss = sim[i]()
        print(f"i = {i}")
        configPlanner = {
            "eta": 0.15,
            "subEta": 0.05,
            "maxIteration": 3000,
            "simulator": ss,
            "nearGoalRadius": None,
            "rewireRadius": None,
            "endIterationID": 1,
            "printDebug": True,
            "localOptEnable": True,
        }

        # global
        a = GlobalSampling(config=configPlanner)
        globals.append(a.avgtime)
        globalsmin.append(a.mintime)
        globalsmax.append(a.maxtime)
        globalsstd.append(a.stdtime)

        # informed
        b = InformedSampling(ss.xs, ss.xg, configPlanner)
        infoms.append(b.avgtime)
        infomsmin.append(b.mintime)
        infomsmax.append(b.maxtime)
        infomsstd.append(b.stdtime)

        # locgap
        c = LocalGapSampling(ss.xs, ss.xg, configPlanner)
        locals.append(c.avgtime)
        localsmin.append(c.mintime)
        localsmax.append(c.maxtime)
        localsstd.append(c.stdtime)

    import matplotlib.pyplot as plt
    from matplotlib import ticker
    import matplotlib

    matplotlib.rcParams.update({"font.size": 12})
    plt.rcParams["svg.fonttype"] = "none"

    scale = 1.4
    aspect = 16 / 9
    size_aspect = scale * 3.40067
    lw = 1
    fig = plt.figure(
        figsize=(size_aspect * aspect, size_aspect), frameon=True, layout="tight"
    )
    ax = plt.subplot(1, 1, 1)

    ax.plot(
        dim,
        globals,
        label="Global",
        linewidth=lw,
        linestyle="solid",
        marker="o",
        color="#558b2f",
    )
    ax.plot(
        dim,
        infoms,
        label="Informed",
        linewidth=lw,
        linestyle="solid",
        marker="o",
        color="#ff8f00",
    )
    ax.plot(
        dim,
        locals,
        label="Local",
        linewidth=lw,
        linestyle="solid",
        marker="o",
        color="#1565c0",
    )

    ax.set_xlabel("System Dimension")
    ax.set_ylabel("Time nanosec")
    ax.grid(axis="y")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=3,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.2,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=1,
        width=1,
        colors="black",
        grid_color="gray",
        grid_alpha=0.1,
    )
    ax.set_xlim((1.9, 6.1))
    # ax.legend(loc=1)
    ax.legend(ncols=4, bbox_to_anchor=(0, 1), loc="lower left")
    plt.show()

    # error bar plot
    # scale = 1.4
    # aspect = 16 / 9
    # size_aspect = scale * 3.40067
    # lw = 1
    # fig = plt.figure(figsize=(size_aspect * aspect, size_aspect), frameon=True, layout="tight")
    # ax = plt.subplot(1, 1, 1)

    # ax.errorbar(dim, globals, yerr=[globalsstd, globalsstd], label="Global", fmt=" ", capsize=5, linewidth=lw, linestyle="solid", color="#558b2f")
    # ax.errorbar(dim, infoms, yerr=[infomsstd, infomsstd], label="Informed", fmt=" ", capsize=5, linewidth=lw, linestyle="solid", color="#ff8f00")
    # ax.errorbar(dim, locals, yerr=[localsstd, localsstd], label="Local", fmt=" ", capsize=5, linewidth=lw, linestyle="solid", color="#1565c0")

    # ax.set_xlabel("System Dimension")
    # ax.set_ylabel("Time sec")
    # ax.grid(axis="y")
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.tick_params(axis="both", which="major", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.2)
    # ax.tick_params(axis="both", which="minor", direction="in", length=1, width=1, colors="black", grid_color="gray", grid_alpha=0.1)
    # ax.set_xlim((1.9, 6.1))
    # # ax.set_ylim((0.0, 1e-9))
    # # ax.legend(loc=1)
    # ax.legend(ncols=4, bbox_to_anchor=(0, 1), loc="lower left")
    # plt.show()
