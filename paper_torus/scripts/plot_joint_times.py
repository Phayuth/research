import matplotlib.pyplot as plt
import numpy as np
from util import np_load, np_save, np_load_csv, option_runner


def times_zero_to_one(numseg):
    time = np.linspace(0, 1, num=numseg)
    return time


def plot_joint_times(joint_path, times, joint_path_aux=None):
    numseg, numjoints = joint_path.shape
    fig, axs = plt.subplots(numjoints, 1, sharex=True)
    for i in range(numjoints):
        axs[i].plot(
            times,
            joint_path[:, i],
            color="blue",
            marker="o",
            linestyle="dashed",
            linewidth=2,
            markersize=6,
            label=f"Position",
        )
    if joint_path_aux is not None:
        for i in range(numjoints):
            axs[i].plot(
                times,
                joint_path_aux[:, i],
                color="orange",
                marker="o",
                linestyle="dashed",
                linewidth=2,
                markersize=6,
                label=f"Position (Aux)",
            )

    # visual setup
    for i in range(numjoints):
        axs[i].set_ylabel(f"Joint {i+1}")
        axs[i].set_xlim(times[0], times[-1])
        axs[i].set_ylim(-np.pi, np.pi)
        axs[i].legend(loc="upper right")
        axs[i].grid(True)
    axs[-1].set_xlabel("Time")
    plt.show()


def plot_joint_so2s():
    joint12 = np_load_csv("paper_so2s_path.csv")
    times = times_zero_to_one(joint12.shape[0])
    plot_joint_times(joint12, times)


def plot_joint_time():
    joint = np_load("collision_free_tour.npy")
    times = times_zero_to_one(joint.shape[0])
    plot_joint_times(joint, times)


if __name__ == "__main__":
    func = [
        plot_joint_so2s,
        plot_joint_time,
    ]
    option_runner(func)
