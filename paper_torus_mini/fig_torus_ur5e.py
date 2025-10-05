import time
import numpy as np

np.random.seed(9)

from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI
from spatial_geometry.utils import Utils

simu = UR5eArmCoppeliaSimAPI()

plantreecfg = {
    "planner": 5,  # 5
    "eta": 0.15,
    "subEta": 0.05,
    "maxIteration": 1000,
    "simulator": simu,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True,
}


xStart = np.array(
    [
        -1.66792136827577,
        -1.95428576091908,
        -1.50018072128295,
        -4.45825996021413,
        -1.65081054369081,
        -3.14131814638246,
    ]
).reshape(6, 1)
xApp = np.array(
    [
        -0.024881664906637013,
        -1.8307167492308558,
        -1.5795478820800781,
        -2.6565920315184535,
        -1.555636231099264,
        -3.134223286305563,
    ]
).reshape(6, 1)
xGoal = np.array(
    [
        -0.0027387777911584976,
        -1.9624139271178187,
        -1.4210033416748047,
        -2.6216727695860804,
        -1.4972699324237269,
        -3.134235207234518,
    ]
).reshape(6, 1)

# pa = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plantreecfg)
# pq = pa.begin_planner()
# pa.plot_performance()
# simu.play_back_path(pq)


XAPPALT = Utils.find_alt_config(xApp, np.array(simu.configLimit), None, False)
XGOALALT = Utils.find_alt_config(xGoal, np.array(simu.configLimit), None, False)

distorg = np.linalg.norm(xApp - xStart)
print(f"> distorg: {distorg}")
distalt = np.linalg.norm((XAPPALT - xStart), axis=0)
print(f"> distalt: {distalt}")


def demo_redun():
    # demonstrate redundancy
    xs = np.array([0.0] * 6).reshape(6, 1)
    for i in range(XAPPALT.shape[1]):
        xa = XAPPALT[:, i, np.newaxis]
        xg = XGOALALT[:, i, np.newaxis]

        simu.set_joint_position(simu.jointDynamicHandles, xs)
        time.sleep(2)
        simu.set_joint_position(simu.jointDynamicHandles, xa)
        print(i, xa)
        time.sleep(3)


def plan_individual():
    # planning and check path 1 by 1
    for i in range(XAPPALT.shape[1]):
        xa = XAPPALT[:, i, np.newaxis]
        xg = XGOALALT[:, i, np.newaxis]
        print(i, xa)

        pa = RRTPlannerAPI.init_normal(xStart, xa, xg, plantreecfg)
        pq = pa.begin_planner()
        time.sleep(3)

        pa.plot_performance()
        simu.play_back_path(pq)
        input("enter to continue")


def plan_one_from_candidate():
    i = 0  # 5.157047081477856
    # i = 1  # 9.59335527471311
    xa = XAPPALT[:, i, np.newaxis]
    xg = XGOALALT[:, i, np.newaxis]
    print(i, xa)

    pa = RRTPlannerAPI.init_normal(xStart, xa, xg, plantreecfg)
    pq = pa.begin_planner()

    pa.plot_performance()
    simu.play_back_path(pq)


def plan_all_candidate_auto():
    pa = RRTPlannerAPI.init_alt_q_torus(xStart, xApp, xGoal, plantreecfg, None)
    pq = pa.begin_planner()

    pa.plot_performance()
    simu.play_back_path(pq)


if __name__ == "__main__":
    # demo_redun()
    # plan_individual()
    # plan_one_from_candidate()
    # plan_all_candidate_auto()

    xStart = np.deg2rad([-100, -70, 100, -120, -90, 0]).reshape(6, 1)
    xApp = np.deg2rad([-165, -70, 100, -30, 90, 0]).reshape(6, 1)
    xApplong = np.deg2rad([-165 + 360, -70, 100, -30 + 360, 90, 0]).reshape(6, 1)
    xGoal = xApplong.copy()

    simu.set_joint_position(simu.jointDynamicHandles, xStart)
    simu.set_joint_position(simu.jointVirtualHandles, xApp)
    # simu.play_back_path(pq)

    # a = np.linspace(xStart.flatten(), xApp.flatten(), num=10, endpoint=True).T
    # print(f"> a.shape: {a.shape}")
    # print(f"> a: {a}")
    # simu.play_back_path(a)

    # a = np.linspace(xStart.flatten(), xApplong.flatten(), num=10, endpoint=True).T
    # print(f"> a.shape: {a.shape}")
    # print(f"> a: {a}")
    # simu.play_back_path(a)

    pa = RRTPlannerAPI.init_normal(xStart, xApplong, xGoal, plantreecfg)
    pq = pa.begin_planner()
    pa.plot_performance()
    # np.save("./ssss.npy", pq)
    # pq = np.load("./ssss.npy")
    # simu.play_back_path(pq)
