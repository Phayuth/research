import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import os

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def make_costgrid(npoints=10, dof=2):
    ngaps = npoints - 1
    costshape = tuple([ngaps] * dof)
    costgrid = np.full(shape=costshape, fill_value=0.0)
    return costgrid


def make_geometric_grid(
    npoints=10,
    dof=2,
    linemin=-np.pi,
    linemax=np.pi,
):
    ngaps = npoints - 1
    line = np.linspace(linemin, linemax, npoints)
    sqrcentershape = tuple([ngaps] * dof + [dof])
    sqrcenter = np.empty(sqrcentershape, dtype=float)

    length = line[1] - line[0]
    for idx in np.ndindex(sqrcenter.shape[:-1]):
        for d in range(dof):
            i = idx + (d,)
            sqrcenter[i] = line[idx[d]] + length / 2.0
    return sqrcenter, length


def is_point_in_ndcube(point, sqrcenter, length, indices):
    for d in range(len(indices)):
        lower = sqrcenter[indices + (d,)] - length / 2.0
        upper = sqrcenter[indices + (d,)] + length / 2.0
        if not (lower <= point[d] <= upper):
            return False
    return True


def which_indices_point_in_ndcube(point, sqrcenter, length):
    pass


def get_2d_rec_mplpatch(sqrcenter, length, color, alpha):
    lower_x = sqrcenter[0] - length / 2.0
    lower_y = sqrcenter[1] - length / 2.0
    rect = patches.Rectangle(
        (lower_x, lower_y),
        length,
        length,
        linewidth=0.5,
        fill=True,
        edgecolor="gray",
        alpha=alpha,
        facecolor=color,
    )
    return rect


def get_2d_text_mplpatch(sqrcenter, text, fontsize=10):
    text_patch = plt.text(
        sqrcenter[0],
        sqrcenter[1],
        text,
        fontsize=fontsize,
        ha="center",
        va="center",
        color="black",
    )
    return text_patch


def __score_cube():
    Point = np.random.uniform(-np.pi, np.pi, size=(500, 2))

    costgrid = make_costgrid(npoints=10, dof=2)
    sqrcenter, length = make_geometric_grid(npoints=10, dof=2)

    cumsum_pointinsquare = np.zeros_like(costgrid)
    cumsum_collision = np.zeros_like(costgrid)

    for i in range(costgrid.shape[0] - 1):
        for j in range(costgrid.shape[0] - 1):
            for p in Point:
                if is_point_in_ndcube(p, sqrcenter, length, (i, j)):
                    cumsum_pointinsquare[i, j] += 1
                    if np.random.rand() < 0.3:
                        cumsum_collision[i, j] += 1

    print("cumsum_collision:\n", cumsum_collision)
    print("cumsum_pointinsquare:\n", cumsum_pointinsquare)
    cellscore = cumsum_collision / cumsum_pointinsquare
    print("cellscore:\n", cellscore)


def __usage():
    print("Example usage of a 2d geometric square grid.")
    sqrcenter, length = make_geometric_grid(npoints=10, dof=2)

    fig, ax = plt.subplots()
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", "box")
    for i in range(sqrcenter.shape[0]):
        for j in range(sqrcenter.shape[1]):
            rect = get_2d_rec_mplpatch(
                sqrcenter[i, j],
                length,
                plt.cm.viridis(i / sqrcenter.shape[0]),
                0.5,
            )
            text = get_2d_text_mplpatch(
                sqrcenter[i, j],
                f"({i},{j})",
                fontsize=8,
            )
            ax.add_patch(rect)
            ax.add_artist(text)
    plt.show()


def __simplepcm():
    pcm = np.load(os.path.join(rsrc, "cspace_grid_cellscore.npy"))

    sqrcenter, length = make_geometric_grid(npoints=10, dof=2)
    fig, ax = plt.subplots()
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", "box")
    for i in range(sqrcenter.shape[0]):
        for j in range(sqrcenter.shape[1]):
            rect = get_2d_rec_mplpatch(
                sqrcenter[i, j],
                length,
                plt.cm.viridis(pcm[i, j]),
                0.8,
            )
            text = get_2d_text_mplpatch(
                sqrcenter[i, j],
                f"{pcm[i, j]:.2f}",
                fontsize=8,
            )
            ax.add_patch(rect)
            ax.add_artist(text)
    plt.show()


if __name__ == "__main__":
    # __score_cube()
    __usage()
    __simplepcm()
