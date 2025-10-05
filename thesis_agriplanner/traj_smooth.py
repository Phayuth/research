import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
import pickle

from trajectory_generator.traj_interpolator import (
    CubicSplineInterpolationIndependant,
    MonotoneSplineInterpolationIndependant,
    BSplineSmoothingUnivariant,
    BSplineInterpolationIndependant,
    SmoothSpline,
)


pikf = "./datasave/planner_thesis_data/initial_to_grasp.pkl"
# pikf = "./datasave/planner_thesis_data/pregrasp_to_drop.pkl"
with open(pikf, "rb") as file:
    data = pickle.load(file)
    print(f"> data.shape: {data.shape}")

vmin = -np.pi
vmax = np.pi
tf = 5
time = np.linspace(0, tf, data.shape[1])
timenew = np.linspace(0, tf, 1001)

# cb = CubicSplineInterpolationIndependant(time, data)
# cb = MonotoneSplineInterpolationIndependant(time, data, 2)
# cb = BSplineInterpolationIndependant(time, data, degree=3)
cb = BSplineSmoothingUnivariant(time, data, smoothc=0.01, degree=5)
# cb = SmoothSpline(time, data, lam=0.001)

p = cb.eval_pose(timenew)
v = cb.eval_velo(timenew)
a = cb.eval_accel(timenew)
if isinstance(p, list):
    p = np.array(p)
    v = np.array(v)
    a = np.array(a)

# position
fig, axs = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for i in range(6):
    axs[i].plot(time, data[i], "k*", label=f"Joint Pos value {i+1}")
    axs[i].plot(timenew, p[i], "b-", label=f"Joint Pos Interp {i+1}")
    axs[i].set_ylabel(f"JPos {i+1}")
    axs[i].set_xlim(time[0], time[-1])
    axs[i].legend(loc="upper right")
    axs[i].grid(True)
axs[-1].set_xlabel("Time")
fig.suptitle("Joint Position")
plt.tight_layout(rect=[0, 0, 1, 0.96])


# velocity
fig1, axs1 = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for i in range(6):
    axs1[i].plot(timenew, v[i], "g-", label=f"Joint Velo Interp {i+1}")
    axs1[i].plot(
        timenew, [vmin] * len(timenew), "m", label=f"Velo min : {vmin:.3f}"
    )
    axs1[i].plot(
        timenew, [vmax] * len(timenew), "c", label=f"Velo max : {vmax:.3f}"
    )
    axs1[i].set_ylabel(f"JVelo {i+1}")
    axs1[i].set_xlim(time[0], time[-1])
    axs1[i].legend(loc="upper right")
    axs1[i].grid(True)
axs1[-1].set_xlabel("Time")
fig1.suptitle("Joint Velo")
plt.tight_layout(rect=[0, 0, 1, 0.96])


# acc
fig2, axs2 = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
for i in range(6):
    axs2[i].plot(timenew, a[i], "r-", label=f"Joint Acc Interp {i+1}")
    axs2[i].set_ylabel(f"JAcc {i+1}")
    axs2[i].set_xlim(time[0], time[-1])
    axs2[i].legend(loc="upper right")
    axs2[i].grid(True)
axs2[-1].set_xlabel("Time")
fig2.suptitle("Joint Acc")
plt.tight_layout(rect=[0, 0, 1, 0.96])


plt.show()
