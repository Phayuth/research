import os
import matplotlib.pyplot as plt
from simulator.sim_planar_rr import RobotArm2DSimulator
import numpy as np
from task_map import PaperICCAS2024


def fig_cspace_2d():
    """
    generate cspace figure with so2(minimal space), extended space.
    switch between torus=true/false for the corresponding
    """
    torus = False
    env = RobotArm2DSimulator(PaperICCAS2024(), torusspace=torus)
    fig, ax = plt.subplots(1, 1)
    colp = env.plot_cspace(ax)
    if torus:
        ax.set_xlim([-2 * np.pi, 2 * np.pi])
        ax.set_ylim([-2 * np.pi, 2 * np.pi])
        ax.axhline(y=np.pi, color="gray", alpha=0.4)
        ax.axhline(y=-np.pi, color="gray", alpha=0.4)
        ax.axvline(x=np.pi, color="gray", alpha=0.4)
        ax.axvline(x=-np.pi, color="gray", alpha=0.4)
    else:
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
    plt.show()

    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    if torus:
        np.save(rsrc + "collisionpoint_exts.npy", colp)
    else:
        np.save(rsrc + "collisionpoint_so2s.npy", colp)


def fig_sequential_task_2d():
    """
    generate and visualize figure with minimal and extended space.
    """
    torus = False
    env = RobotArm2DSimulator(PaperICCAS2024(), torusspace=torus)
    task0 = np.array([-np.pi / 2, 0.0])  # initial configuration
    task1 = np.array([-1.0, 2.0])  # grab cup
    task2 = np.array([np.pi + 1, -2.0])  # place cup
    thetas = np.vstack((task0, task1, task2)).T
    env.plot_view(thetas)


if __name__ == "__main__":
    # fig_cspace_2d()
    fig_sequential_task_2d()
