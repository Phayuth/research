import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import pickle
import pandas as pd


path = "./datasave/planner_thesis_data/"

a = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_2_withlocgap_False_costgraph.pkl"
)
b = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_4_withlocgap_False_costgraph.pkl"
)
c = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_15_withlocgap_False_costgraph.pkl"
)
d = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_5_withlocgap_False_costgraph.pkl"
)
e = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_6_withlocgap_False_costgraph.pkl"
)

f = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_4_withlocgap_True_costgraph.pkl"
)
g = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_15_withlocgap_True_costgraph.pkl"
)
h = (
    path
    + "env_TaskSpace2DSimulator_types_Single_planner_5_withlocgap_True_costgraph.pkl"
)

matplotlib.rcParams.update({"font.size": 12})
plt.rcParams["svg.fonttype"] = "none"


def load_cost_graph_pkl(path):
    with open(path, "rb") as file:
        loadedList = pickle.load(file)
    return loadedList


def mean_curve(loadedList):
    arrs = [np.array(list) for list in loadedList]
    dfs = [
        pd.DataFrame(
            ar[:, 1], index=ar[:, 0].astype(np.int32), columns=[f"cost-{i}th"]
        )
        for i, ar in enumerate(arrs)
    ]

    newIdx = pd.Index(range(2000))
    newDfs = [d.reindex(newIdx) for d in dfs]

    mergedf = pd.concat(newDfs, axis=1)
    mergedf = mergedf.interpolate(method="values", limit_direction="both")

    # an.to_csv("out.csv", index=True)
    meandfs = mergedf.mean(axis=1)
    return meandfs


def get_(path):
    ll = load_cost_graph_pkl(path)
    return mean_curve(ll)


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

ax.plot(
    meandf_inf.index,
    meandf_inf.values,
    label="Infm-RRT*",
    linewidth=lw,
    linestyle="solid",
    color="#558b2f",
)
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


# 6.759699575150458 coptimal
ax.plot(
    meandf_cntlocgap.index,
    [6.759699575150458] * len(meandf_cntlocgap.index),
    linewidth=lw,
    linestyle=":",
    color="k",
)

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
ax.set_xlim((0, 2025))
# ax.legend(loc=1)
ax.legend(ncols=4, bbox_to_anchor=(0, 1), loc="lower left")
plt.show()
