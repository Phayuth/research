import numpy as np
import matplotlib.pyplot as plt
from geometric_ellipse import *
import os

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))


def bulk_collisioncheck(Xrand):
    # dummy collision check function
    return np.array([False] * Xrand.shape[0])


# q1 = np.array([-0.5] * 2).reshape(-1, 1)
# q2 = np.array([0.5] * 2).reshape(-1, 1)
q1 = np.array([-1.0, 2.5]).reshape(-1, 1)
q2 = np.array([1.0, 2.5]).reshape(-1, 1)
q3 = np.array([0.15, 0.60]).reshape(-1, 1)
q4 = np.array([2.5, 1.5]).reshape(-1, 1)
q5 = np.array([-2.5, -1.5]).reshape(-1, 1)
q6 = np.array([2.40, -0.4]).reshape(-1, 1)
q7 = np.array([-2.0, 2.5]).reshape(-1, 1)
q8 = np.array([1.0, -2.0]).reshape(-1, 1)
q9 = np.array([-3.0, 0.0]).reshape(-1, 1)
q10 = np.array([-3.0, 2.5]).reshape(-1, 1)

qs = q3
qg = q4

if __name__ == "__main__":
    cmin = np.linalg.norm(qg - qs)
    cMaxguess = 1.47 * cmin

    Xinf_bulk = informed_sampling_bulk(qs, qg, cMaxguess, 1000)
    Xinfinside = informed_surface_sampling_bulk(qs, qg, cMaxguess, 1000)

    la1 = cMaxguess / 2
    sa1 = cMaxguess / 2
    Xb1 = custom_surface_sampling(qs, qg, la1, sa1, 1000)
    Xi1 = custom_inside_sampling(qs, qg, la1, sa1, 1000)

    la2 = cMaxguess / 2
    sa2 = la2*1.9
    Xb2 = custom_surface_sampling(qs, qg, la2, sa2, 1000)
    Xi2 = custom_inside_sampling(qs, qg, la2, sa2, 1000)

    la3 = cmin/2
    sa3 = cmin/2
    Xb3 = custom_surface_sampling(qs, qg, la3, sa3, 1000)
    Xi3 = custom_inside_sampling(qs, qg, la3, sa3, 1000)

    # Xcol_bulk = bulk_collisioncheck(Xrand_bulk)
    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    ax.scatter(Xinfinside[:, 0], Xinfinside[:, 1], s=5, c="b", label="informed")
    ax.scatter(Xb1[:, 0], Xb1[:, 1], s=5, c="g", label="equals")
    ax.scatter(Xb2[:, 0], Xb2[:, 1], s=5, c="r", label="case1")
    ax.scatter(Xb3[:, 0], Xb3[:, 1], s=5, c="m", label="case2")
    ax.scatter(qs[0], qs[1], s=50, c="k", marker="x")
    ax.scatter(qg[0], qg[1], s=50, c="k", marker="x")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
    plt.show()
