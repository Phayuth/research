import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import pickle
import matplotlib.pyplot as plt
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI

from trajectory_generator.traj_interpolator import (
    CubicSplineInterpolationIndependant,
    MonotoneSplineInterpolationIndependant,
    BSplineSmoothingUnivariant,
    BSplineInterpolationIndependant,
    SmoothSpline,
)

# real experiment motion play back in the simulator

pikf = "./datasave/planner_thesis_data/initial_to_grasp.pkl"
# pikf = "./datasave/planner_thesis_data/pregrasp_to_drop.pkl"
with open(pikf, "rb") as file:
    data = pickle.load(file)
    print(f"> data.shape: {data.shape}")

# tf = 6
# vmin = -np.pi
# vmax = np.pi
# time = np.linspace(0, tf, data.shape[1])
# # cb = CubicSplineInterpolationIndependant(time, data)
# # cb = MonotoneSplineInterpolationIndependant(time, data, 2)
# # cb = BSplineInterpolationIndependant(time, data, degree=3)
# cb = BSplineSmoothingUnivariant(time, data, smoothc=0.01, degree=5)
# # cb = SmoothSpline(time, data, lam=0.001)

# timenew = np.linspace(0, tf, 1001)
# p = cb.eval_pose(timenew)
# v = cb.eval_velo(timenew)
# a = cb.eval_accel(timenew)

# if isinstance(p, list):
#     p = np.array(p)
#     v = np.array(v)
#     a = np.array(a)

simu = UR5eArmCoppeliaSimAPI()
ind = -1  # 0, 29, 45, -1
jj = data[:, ind, np.newaxis]
simu.set_joint_position(simu.jointDynamicHandles, jj)
# simu.play_back_path(p)
