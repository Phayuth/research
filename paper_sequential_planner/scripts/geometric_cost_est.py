import numpy as np
import matplotlib.pyplot as plt
from geometric_ellipse import *


np.random.seed(42)


def sample_Xstartgoal(xcenter, Rbest, cmin, dof=2):
    if cmin > 2 * Rbest:
        raise ValueError("cmin cannot be larger than 2*Rbest")

    Rrand = sample_rotation_matrix(dof)
    cMax = 2 * Rbest
    L = hyperellipsoid_informed_axis_length(cMax, cmin, dof)
    e1 = np.zeros((dof, 1))
    e1[0, 0] = 1.0
    u = Rrand @ e1
    rmin = cmin / 2
    qa = xcenter - rmin * u
    qb = xcenter + rmin * u
    return qa, qb


def _sample_X():
    xcent = sampling_circle()
    Rbest = 1.0
    cMax = 2 * Rbest
    cmin = 1.0
    qa, qb = sample_Xstartgoal(xcent.reshape(-1, 1), Rbest, cmin)
    XRand = np.empty((10000, 2))
    for i in range(10000):
        R = rotation_to_world(qa.reshape(-1, 1), qb.reshape(-1, 1))
        xrand = informed_sampling(xcent.reshape(-1, 1), cMax, cmin, R)
        XRand[i, :] = xrand.flatten()

    fig, ax = plt.subplots()
    ax.plot(XRand[:, 0], XRand[:, 1], "ro", markersize=2)
    e = get_2d_ellipse_informed_mplpatch(
        qa.reshape(-1, 1), qb.reshape(-1, 1), cMax=2 * Rbest, cMin=cmin
    )
    ax.add_patch(e)
    cir = get_circle_mplpatch(xcent, r=Rbest)
    ax.add_patch(cir)
    ax.plot(qa[0], qa[1], "bo", label="qa")
    ax.plot(qb[0], qb[1], "go", label="qb")
    ax.plot(xcent[0], xcent[1], "ko", label="xcenter")
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.legend()
    plt.show()


def _recover_q_from_center_and_rotation():
    qa = np.array([0.0, 0.0]).reshape(-1, 1)
    qb = np.array([1.0, 1.0]).reshape(-1, 1)
    cMin = np.linalg.norm(qb - qa)
    RRR = rotation_to_world(qa, qb)
    qcenter = (qa + qb) / 2
    e1 = np.zeros((2, 1))
    e1[0, 0] = 1.0
    u = RRR @ e1
    d = cMin / 2
    qarecover = qcenter - d * u
    qbrecover = qcenter + d * u


def three_point_ellipse_sampling(qs, qe, qm):
    p1 = np.linspace(qs, qm, num=10)
    p2 = np.linspace(qm, qe, num=10)
    p = np.vstack((p1, p2))
    return p


dof = 2
Rbest = 1.0
cMax = 2 * Rbest
cmin = 1.0
n = 4
shape = (n, dof + dof)
Qcircle = np.empty(shape=(n, dof))
Qrand = np.empty(shape=shape)
for i in range(n):
    qcenter = sampling_circle(Rbest)
    Qcircle[i, :] = qcenter
    qa, qb = sample_Xstartgoal(qcenter.reshape(-1, 1), Rbest, cmin, dof=dof)
    Qrand[i, 0:dof] = qa.flatten()
    Qrand[i, dof : dof + dof] = qb.flatten()


fig, ax = plt.subplots()
for i in range(n):
    qcenter = Qcircle[i, :]
    qa = Qrand[i, 0:dof]
    qb = Qrand[i, dof : dof + dof]
    rot = rotation_to_world(qa.reshape(-1, 1), qb.reshape(-1, 1))
    qm = elliptical_sampling(
        qa.reshape(-1, 1), qb.reshape(-1, 1), rot, unit_ball_surface_sampling
    )
    p = three_point_ellipse_sampling(qa, qb, qm)

    ax.plot(qcenter[0], qcenter[1], "bo", label="q_center" if i == 0 else "")
    ax.plot(qa[0], qa[1], "ro", label="q_a" if i == 0 else "")
    ax.plot(qb[0], qb[1], "go", label="q_b" if i == 0 else "")
    ax.plot(qm[0], qm[1], "mo", label="q_mid" if i == 0 else "")
    ax.plot(
        p[:, 0],
        p[:, 1],
        "k--",
        linewidth=0.5,
        label="ellipse arc" if i == 0 else "",
    )

    e = get_2d_ellipse_informed_mplpatch(
        qa.reshape(-1, 1), qb.reshape(-1, 1), cMax=2 * Rbest, cMin=cmin
    )
    ax.add_patch(e)
    c = get_circle_mplpatch(qcenter, r=Rbest)
    ax.add_patch(c)

ax.set_aspect("equal", "box")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.grid(True)
ax.legend()
plt.show()
