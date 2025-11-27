import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)

# # setup grid
# npoints = 10
# dof = 2
# ngaps = npoints - 1
# line = np.linspace(-np.pi, np.pi, npoints)
# print(line.shape)

# costshape = tuple([ngaps] * dof)
# costgrid = np.full(shape=costshape, fill_value=0.0)
# print("costgrid.shape:", costgrid.shape)

# sqrtcentershape = tuple([ngaps] * dof + [dof])
# sqrcenter = np.empty(sqrtcentershape, dtype=float)
# print("sqrcenter.shape:", sqrcenter.shape)

# length = line[1] - line[0]
# # for idx in np.ndindex(sqrcenter.shape[:-1]):
# #     print("idx:", idx)

# for i in range(ngaps):
#     for j in range(ngaps):
#         print(f"i:{i}, j:{j}")
#         sqrcenter[i, j, 0] = line[i] + length / 2.0
#         sqrcenter[i, j, 1] = line[j] + length / 2.0


def is_point_in_square(point, sqrcenter, length, i, j):
    lower_x = sqrcenter[i, j, 0] - length / 2.0
    upper_x = sqrcenter[i, j, 0] + length / 2.0
    lower_y = sqrcenter[i, j, 1] - length / 2.0
    upper_y = sqrcenter[i, j, 1] + length / 2.0
    if lower_x <= point[0] <= upper_x and lower_y <= point[1] <= upper_y:
        return True
    else:
        return False


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


# def _plot_grid():
#     freecolor = "white"
#     colcolor = "red"
#     cmap = LinearSegmentedColormap.from_list("custom_cmap", [freecolor, colcolor])

#     Rect = []
#     fig, ax = plt.subplots()
#     # ax.plot(Point[:, 0], Point[:, 1], "ko", markersize=2)
#     for i in range(ngaps):
#         for j in range(ngaps):
#             rect = _make_rect_patches(i, j, cmap)
#             Rect.append(rect)
#             ax.add_patch(rect)
#             # text = f"({i},{j}) {cellscore[i, j]:.2f}"
#             # ax.text(
#             #     sqrcenter[i, j, 0] - length / 2.0 + 0.1,
#             #     sqrcenter[i, j, 1] - length / 2.0 + 0.1,
#             #     text,
#             #     fontsize=6,
#             # )
#     ax.set_xlim(-np.pi, np.pi)
#     ax.set_ylim(-np.pi, np.pi)
#     ax.set_aspect("equal", "box")
#     plt.show()
