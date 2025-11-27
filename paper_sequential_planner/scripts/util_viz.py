import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis


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


def plot_tf(T, names):
    ax = make_3d_axis(ax_s=1, unit="m")
    plot_transform(ax=ax, s=0.5, name="base_frame")  # basis
    if isinstance(T, list):
        for i, t in enumerate(T):
            if names is not None:
                name = names[i]
            else:
                name = f"{i}"
            plot_transform(ax=ax, A2B=t, s=0.1, name=name)
    else:
        plot_transform(ax=ax, A2B=T, s=0.1, name="frame")
    return ax


def plot_tf_tour(T, names, tour):
    ax = plot_tf(T, names)
    if tour is not None:
        for i, j in tour:
            ax.plot(
                [T[i][0, 3], T[j][0, 3]],
                [T[i][1, 3], T[j][1, 3]],
                [T[i][2, 3], T[j][2, 3]],
                color="red",
                linewidth=2,
            )
    ax.set_title(f"Tour: {tour}")
    return ax


def plot_2d_tour(coords, tour):
    """
    coords: (n,2) array of xy points
    tour: (n,) array of city indices
    """
    ordered_coords = coords[tour]

    # close the loop
    closed_coords = np.vstack([ordered_coords, ordered_coords[0]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(closed_coords[:, 0], closed_coords[:, 1], "-o", markersize=8)
    for i, (x, y) in enumerate(coords):
        ax.text(x + 0.02, y + 0.02, str(i), fontsize=9)
    return ax


def plot_2d_tour_coord(coords, tour):
    """for tsp"""
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, (x, y) in coords.items():
        ax.plot(x, y, "bo")
        ax.text(x + 0.2, y + 0.2, str(i), fontsize=12)

    for i, j in tour:
        xi, yi = coords[i]
        xj, yj = coords[j]
        ax.plot([xi, xj], [yi, yj], "r-")

    ax.set_title(f"Tour: {tour}")
    ax.axis("equal")
