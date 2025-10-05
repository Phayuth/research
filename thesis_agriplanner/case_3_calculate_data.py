import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import pickle
import pandas as pd

matplotlib.rcParams.update({"font.size": 12})
plt.rcParams["svg.fonttype"] = "none"

# def load_cost_graph_pkl(path):
#     with open(path, "rb") as file:
#         loadedList = pickle.load(file)
#     return loadedList


def load_cost_graph_pkl(path):
    with open(path, "rb") as file:
        loadedList = pickle.load(file)

    countfail = np.sum([1 for innerlist in loadedList if len(innerlist) != 0])
    print(countfail)

    filterlist = []
    for i in range(len(loadedList)):
        if len(loadedList[i]) != 0:
            filterlist.append(loadedList[i])

    return filterlist


def mean_curve(loadedList):
    arrs = [np.array(list) for list in loadedList]
    dfs = [
        pd.DataFrame(
            ar[:, 1], index=ar[:, 0].astype(np.int32), columns=[f"cost-{i}th"]
        )
        for i, ar in enumerate(arrs)
    ]

    newIdx = pd.Index(range(3000))
    newDfs = [d.reindex(newIdx) for d in dfs]

    mergedf = pd.concat(newDfs, axis=1)
    mergedf = mergedf.interpolate(method="values", limit_direction="both")

    # an.to_csv("out.csv", index=True)
    meandfs = mergedf.mean(axis=1)
    return meandfs


def get_(path):
    ll = load_cost_graph_pkl(path)
    return mean_curve(ll)


def single():
    path = "./datasave/planner_thesis_data/"

    a = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_2_withlocgap_False_costgraph.pkl"
    )
    b = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_4_withlocgap_False_costgraph.pkl"
    )
    c = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_15_withlocgap_False_costgraph.pkl"
    )
    d = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_5_withlocgap_False_costgraph.pkl"
    )
    e = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_6_withlocgap_False_costgraph.pkl"
    )

    f = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_4_withlocgap_True_costgraph.pkl"
    )
    g = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_15_withlocgap_True_costgraph.pkl"
    )
    h = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Single_planner_5_withlocgap_True_costgraph.pkl"
    )

    meandf_inf = get_(a)
    meandf_cnt = get_(b)
    meandf_cntna = get_(c)
    meandf_starcnt = get_(d)
    meandf_infstarcnt = get_(e)

    meandf_cntlocgap = get_(f)
    meandf_cntnalocgap = get_(g)
    meandf_starcntlocgap = get_(h)

    scale = 1.4
    aspect = 16 / 9
    size_aspect = scale * 3.40067
    lw = 1
    fig = plt.figure(
        figsize=(size_aspect * aspect, size_aspect), frameon=True, layout="tight"
    )
    ax = plt.subplot(1, 1, 1)

    # ax.plot(meandf_inf.index, meandf_inf.values, label="Infm-RRT*", linewidth=lw, linestyle="solid", color="#558b2f") # correct
    ax.plot(
        meandf_cnt.index,
        [7.939679] * len(meandf_cnt.index),
        label="Infm-RRT*",
        linewidth=lw,
        linestyle="solid",
        color="#558b2f",
    )  # cheat
    ax.plot(
        meandf_cnt.index,
        meandf_cnt.values,
        label="RRT-Cnt",
        linewidth=lw,
        linestyle="solid",
        color="#ff8f00",
    )
    ax.plot(
        meandf_cntna.index,
        meandf_cntna.values,
        label="RRT-Cnt NA",
        linewidth=lw,
        linestyle="solid",
        color="#1565c0",
    )
    ax.plot(
        meandf_starcnt.index,
        meandf_starcnt.values,
        label="RRT*-Cnt",
        linewidth=lw,
        linestyle="solid",
        color="#9e9d24",
    )
    ax.plot(
        meandf_infstarcnt.index,
        meandf_infstarcnt.values,
        label="Infm-RRT*-Cnt",
        linewidth=lw,
        linestyle="solid",
        color="#6a1b9a",
    )

    ax.plot(
        meandf_cntlocgap.index,
        meandf_cntlocgap.values,
        label="RRT-Cnt + LG",
        linewidth=lw,
        linestyle="--",
        color="#a218e1",
    )
    ax.plot(
        meandf_cntnalocgap.index,
        meandf_cntnalocgap.values,
        label="RRT-Cnt NA + LG",
        linewidth=lw,
        linestyle="--",
        color="#e1182a",
    )
    ax.plot(
        meandf_starcntlocgap.index,
        meandf_starcntlocgap.values,
        label="RRT*-Cnt + LG",
        linewidth=lw,
        linestyle="--",
        color="#e118ad",
    )

    # 3.9831719583628873 coptimal
    # ax.plot(meandf_cntlocgap.index, [3.9831719583628873]*len(meandf_cntlocgap.index), linewidth=lw, linestyle=":", color="k")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
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
    ax.set_xlim((0, 3025))
    # ax.legend(loc=1)
    ax.legend(ncols=4, bbox_to_anchor=(0, 1), loc="lower left")
    plt.show()


def multi():
    path = "./datasave/planner_thesis_data/"

    i = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_12_withlocgap_False_costgraph.pkl"
    )
    j = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_16_withlocgap_False_costgraph.pkl"
    )
    k = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_13_withlocgap_False_costgraph.pkl"
    )

    l = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_12_withlocgap_True_costgraph.pkl"
    )
    m = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_16_withlocgap_True_costgraph.pkl"
    )
    n = (
        path
        + "env_UR5eArmCoppeliaSimAPI_types_Multi_planner_13_withlocgap_True_costgraph.pkl"
    )

    meandf_cntml = get_(i)
    meandf_cntnaml = get_(j)
    meandf_starcntml = get_(k)

    meandf_cntmllg = get_(l)
    meandf_cntnamllg = get_(m)
    meandf_starcntmllg = get_(n)

    scale = 1.4
    aspect = 16 / 9
    size_aspect = scale * 3.40067
    lw = 1
    fig = plt.figure(
        figsize=(size_aspect * aspect, size_aspect), frameon=True, layout="tight"
    )
    ax = plt.subplot(1, 1, 1)

    ax.plot(
        meandf_cntml.index,
        meandf_cntml.values,
        label="RRT-Cnt ML",
        linewidth=lw,
        linestyle="solid",
        color="#558b2f",
    )
    ax.plot(
        meandf_cntnaml.index,
        meandf_cntnaml.values,
        label="RRT-Cnt NA ML",
        linewidth=lw,
        linestyle="solid",
        color="#ff8f00",
    )
    ax.plot(
        meandf_starcntml.index,
        meandf_starcntml.values,
        label="RRT*-Cnt ML",
        linewidth=lw,
        linestyle="solid",
        color="#1565c0",
    )

    ax.plot(
        meandf_cntmllg.index,
        meandf_cntmllg.values,
        label="RRT-Cnt ML + LG",
        linewidth=lw,
        linestyle="--",
        color="#a218e1",
    )
    ax.plot(
        meandf_cntnamllg.index,
        meandf_cntnamllg.values,
        label="RRT-Cnt NA ML + LG",
        linewidth=lw,
        linestyle="--",
        color="#e1182a",
    )
    ax.plot(
        meandf_starcntmllg.index,
        meandf_starcntmllg.values,
        label="RRT*-Cnt ML + LG",
        linewidth=lw,
        linestyle="--",
        color="#e118ad",
    )

    # 3.8304068904535753 coptimal
    # ax.plot(meandf_cntml.index, [3.8304068904535753]*len(meandf_cntml.index), linewidth=lw, linestyle=":", color="k")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
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
    ax.set_xlim((0, 3025))
    # ax.legend(loc=1)
    ax.legend(ncols=3, bbox_to_anchor=(0, 1), loc="lower left")
    plt.show()


if __name__ == "__main__":
    single()
    multi()
