import matplotlib.pyplot as plt
import numpy as np

joint12 = np.loadtxt(
    "_projects/_paper_mobile_manipulator/build/paper_mm_simple_path.csv",
    delimiter=",",
)


numseg, numjoints = joint12.shape
time = np.linspace(0, 1, num=numseg)
fig, axs = plt.subplots(numjoints, 1, figsize=(10, 15), sharex=True)
for i in range(numjoints):
    axs[i].plot(
        time,
        joint12[:, i],
        color="blue",
        marker="o",
        linestyle="dashed",
        linewidth=2,
        markersize=6,
        label=f"Joint Pos {i+1}",
    )
    axs[i].set_ylabel(f"JPos {i+1}")
    axs[i].set_xlim(time[0], time[-1])
    axs[i].set_ylim(-np.pi, np.pi)
    axs[i].legend(loc="upper right")
    axs[i].grid(True)
axs[-1].set_xlabel("Time")
fig.suptitle("Joint Position")
plt.show()
