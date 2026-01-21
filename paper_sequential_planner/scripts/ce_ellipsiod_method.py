import numpy as np
import matplotlib.pyplot as plt
from geometric_ellipse import *


def bulk_collisioncheck(Xrand):
    # dummy collision check function
    return np.array([False] * Xrand.shape[0])



if __name__ == "__main__":
    q1 = np.array([-0.5] * 2).reshape(-1, 1)
    q2 = np.array([0.5] * 2).reshape(-1, 1)
    la = 2.0 / 2
    sa = 2
    Xrand_bulk = custom_surface_sampling(q1, q2, la, sa, 1000)
    Xcol_bulk = bulk_collisioncheck(Xrand_bulk)
    print(Xcol_bulk.shape)
    # print(Xcol_bulk)

    Xrand_bulk2 = custom_surface_sampling(q1, q2, la, sa + 1, 1000)
    Xrand_inside = custom_inside_sampling(q1, q2, la, sa, 10000)

    fig, ax = plt.subplots()
    ax.scatter(Xrand_bulk[:, 0], Xrand_bulk[:, 1], s=2, c="g", label="bulk")
    ax.scatter(Xrand_bulk2[:, 0], Xrand_bulk2[:, 1], s=2, c="b", label="bulk2")
    ax.scatter(Xrand_inside[:, 0], Xrand_inside[:, 1], s=2, c="r", label="inside")
    ellipse = patches.Ellipse(
        xy=(0, 0),
        width=2 * la,
        height=2 * sa,
        angle=45,
        edgecolor="r",
        facecolor="none",
    )
    # ax.add_patch(ellipse)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
    plt.show()
