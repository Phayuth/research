import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def informed_sampling(xCenter, cMax, cMin, rotationAxisC):
    L = hyperellipsoid_axis_length(cMax, cMin)
    xBall = unit_ball_sampling()
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand


def hyperellipsoid_axis_length(cMax, cMin, dof=2):  # L
    r1 = cMax / 2
    ri = np.sqrt(cMax**2 - cMin**2) / 2
    diagTerm = [r1] + [ri] * (dof - 1)
    return np.diag(diagTerm)


def unit_ball_sampling(dof=2):
    u = np.random.normal(0.0, 1.0, (dof + 2, 1))
    norm = np.linalg.norm(u)
    u = u / norm
    return u[:dof, :]  # The first N coordinates are uniform in a unit N ball


def rotation_to_world(xStart, xGoal):  # C
    dof = xStart.shape[0]
    cMin = distance_between_config(xStart, xGoal)
    a1 = (xGoal - xStart) / cMin
    I1 = np.array([1.0] + [0.0] * (dof - 1)).reshape(1, -1)
    M = a1 @ I1
    U, _, V_T = np.linalg.svd(M, True, True)
    middleTerm = [1.0] * (dof - 1) + [np.linalg.det(U) * np.linalg.det(V_T.T)]
    return U @ np.diag(middleTerm) @ V_T


def distance_between_config(xFrom, xTo):
    return np.linalg.norm(xTo - xFrom)


def get_2d_ellipse_mplpatch(xStart, xGoal, cMax, cMin):
    L = hyperellipsoid_axis_length(cMax, cMin)
    xCenter = (xStart + xGoal) / 2
    C = rotation_to_world(xStart, xGoal)  # hyperellipsoid rotation axis
    w = L[0, 0] * 2
    h = L[1, 1] * 2
    a = np.degrees(np.arctan2(C[1, 0], C[0, 0]))

    el = patches.Ellipse(
        (xCenter[0], xCenter[1]),
        width=w,
        height=h,
        angle=a,
        fill=False,
        # edgecolor="red",
        linewidth=2,
    )
    return el


def __usage():
    dof = 2  # degrees of freedom for the configuration space
    xStart = np.array([0.0] * dof).reshape(-1, 1)
    xGoal = np.array([1.0] * dof).reshape(-1, 1)
    xCenter = (xStart + xGoal) / 2
    C = rotation_to_world(xStart, xGoal)  # hyperellipsoid rotation axis
    cMin = distance_between_config(xStart, xGoal)
    cMax = 2.0
    XRAND = [informed_sampling(xCenter, cMax, cMin, C) for _ in range(1000)]

    # plot
    fig, (ax, ay) = plt.subplots(1, 2)

    # fig1
    ax.plot(xStart[0], xStart[1], marker="o", color="blue", label="Start")
    ax.plot(xGoal[0], xGoal[1], marker="o", color="green", label="Goal")
    ax.plot(xCenter[0], xCenter[1], marker="x", color="black", label="Center")
    for x in XRAND:
        ax.plot(x[0], x[1], marker=".", color="gray", alpha=0.5)
    el = get_2d_ellipse_mplpatch(xStart, xGoal, cMax, cMin)
    ax.add_patch(el)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)
    ax.legend()

    # fig2
    ay.plot(xStart[0], xStart[1], marker="o", color="blue", label="Start")
    ay.plot(xGoal[0], xGoal[1], marker="o", color="green", label="Goal")
    ay.plot(xCenter[0], xCenter[1], marker="x", color="black", label="Center")
    CMAX = np.linspace(cMin, cMin * 5, 10)
    for cmax in CMAX:
        el = get_2d_ellipse_mplpatch(xStart, xGoal, cmax, cMin)
        ay.add_patch(el)
        fs = f"cMax is +{((cmax - cMin) / cMin) * 100:.1f}% of cMin"
        ay.plot([], [], label=fs)  # dummy plot for legend
    ay.set_aspect("equal", "box")
    ay.set_xlim(-np.pi, np.pi)
    ay.set_ylim(-np.pi, np.pi)
    ay.grid(True)
    ay.legend()
    plt.show()


if __name__ == "__main__":
    __usage()
