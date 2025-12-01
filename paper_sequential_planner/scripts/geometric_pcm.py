import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def make_costgrid(npoints=10, dof=2):
    ngaps = npoints - 1
    costshape = tuple([ngaps] * dof)
    costgrid = np.full(shape=costshape, fill_value=0.0)
    return costgrid


def make_geometric_sqrgrid(
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


# Point = np.random.uniform(-np.pi, np.pi, size=(500, 2))
# cumsum_pointinsquare = np.zeros_like(costgrid)
# cumsum_collision = np.zeros_like(costgrid)


# for p in Point:
#     for i in range(ngaps):
#         for j in range(ngaps):
#             if is_point_in_square(p, i, j):
#                 cumsum_pointinsquare[i, j] += 1
#                 if np.random.rand() < 0.3:
#                     cumsum_collision[i, j] += 1

# print("cumsum_collision:\n", cumsum_collision)
# print("cumsum_pointinsquare:\n", cumsum_pointinsquare)

# cellscore = cumsum_collision / cumsum_pointinsquare
# print("cellscore:\n", cellscore)


def _make_rect_patches(sqrcenter, length, i, j, cmap, cellscore):
    lower_x = sqrcenter[i, j, 0] - length / 2.0
    lower_y = sqrcenter[i, j, 1] - length / 2.0
    rect = patches.Rectangle(
        (lower_x, lower_y),
        length,
        length,
        linewidth=0.5,
        fill=True,
        edgecolor="gray",
        alpha=cellscore,
        facecolor=cmap(cellscore),
    )
    return rect


if __name__ == "__main__":
    sqrcenter, length = make_geometric_sqrgrid(npoints=10, dof=2)

    fig, ax = plt.subplots()
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", "box")

    for i in range(sqrcenter.shape[0]):
        for j in range(sqrcenter.shape[1]):
            rect = _make_rect_patches(sqrcenter, length, i, j, plt.cm.viridis, 0.5)
            ax.add_patch(rect)
    plt.show()
