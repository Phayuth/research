import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import time
import numpy as np

np.random.seed(9)
import matplotlib.pyplot as plt

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI
import pickle

simu = UR5eArmCoppeliaSimAPI()

copsimConfigDualTree = {
    "planner": 5,
    "eta": 0.15,
    "subEta": 0.05,
    "maxIteration": 3000,
    "simulator": simu,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True,
}


# plan to grasp
# xStart = np.array([-1.66792136827577, -1.95428576091908, -1.50018072128295, -4.45825996021413, -1.65081054369081, -3.14131814638246]).reshape(6, 1)
# xApp = np.array([-0.024881664906637013, -1.8307167492308558, -1.5795478820800781, -2.6565920315184535, -1.555636231099264, -3.134223286305563]).reshape(6,1)
# xGoal = np.array([-0.0027387777911584976, -1.9624139271178187, -1.4210033416748047, -2.6216727695860804, -1.4972699324237269, -3.134235207234518]).reshape(6,1)

# drop off
xStart = np.array(
    [
        -0.024881664906637013,
        -1.8307167492308558,
        -1.5795478820800781,
        -2.6565920315184535,
        -1.555636231099264,
        -3.134223286305563,
    ]
).reshape(6, 1)
xApp = np.array(
    [
        -2.306183640156881,
        -1.6935316524901332,
        -1.7450757026672363,
        -4.381070991555685,
        -1.5936411062823694,
        -3.1308134237872522,
    ]
).reshape(6, 1)
xGoal = np.array(
    [
        -2.306183640156881,
        -1.6935316524901332,
        -1.7450757026672363,
        -4.381070991555685,
        -1.5936411062823694,
        -3.1308134237872522,
    ]
).reshape(6, 1)


pa = RRTPlannerAPI.init_warp_q_to_pi(xStart, xApp, xGoal, copsimConfigDualTree)
timestart = time.perf_counter_ns()
pq = pa.planning()
timestop = time.perf_counter_ns()
timetotal = (timestop - timestart) * 1e-9
print(
    f"> timetotal: {timetotal}"
)  # stop at [5.381543524535372]  [5.697273765739118]]
time.sleep(3)

# fileName = f"./initial_to_grasp.pkl"
# fileName = f"./pregrasp_to_drop.pkl"
# with open(fileName, "wb") as file:
#     pickle.dump(pq, file)

# sss = UR5eArmCoppeliaSimAPI() # must create new instance. problem with zmq (deadlock) probably with collision check request. (If I let planning run out of iteration. It worked fine)
# simu.play_back_path(pq)

pa.plot_performance()
