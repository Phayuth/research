import numpy as np
import matplotlib.pyplot as plt
from geometric_ellipse import get_2d_ellipse_mplpatch

np.random.seed(42)


def sampling_twopoints(eta=1.0):
    qa = np.random.uniform(low=-np.pi, high=np.pi, size=(2,))
    qb = np.random.uniform(low=-np.pi, high=np.pi, size=(2,))
    qc = qa + eta * (qa - qb) / np.linalg.norm(qa - qb)
    l = np.linalg.norm(qa - qc)
    return qa, qc, l


def sampling_circle(roffset=0.5):
    limit = np.array(
        [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
        ]
    )
    limit_offset = limit.copy()
    limit_offset[:, 0] += roffset
    limit_offset[:, 1] -= roffset
    qcenter = np.random.uniform(
        low=limit_offset[:, 0], high=limit_offset[:, 1], size=(2,)
    )
    return qcenter


def get_circle_mplpatch(qcenter, roffset=0.5):
    circle = plt.Circle(
        (qcenter[0], qcenter[1]),
        roffset,
        color="b",
        fill=False,
        linestyle="--",
    )
    return circle


dof = 2
n = 1000
shape = (n, dof + dof + 1)
Qrand = np.empty(shape=shape)
for i in range(n):
    qarand, qbrand, lrand = sampling_twopoints()
    Qrand[i, 0:dof] = qarand
    Qrand[i, dof : dof + dof] = qbrand
    Qrand[i, -1] = lrand

Qcircle = np.empty(shape=(n, dof))
for i in range(n):
    qcenter = sampling_circle()
    Qcircle[i, :] = qcenter


fig, ax = plt.subplots()
for i in range(n):
    qarand = Qrand[i, 0:dof]
    qbrand = Qrand[i, dof : dof + dof]
    lrand = Qrand[i, -1]
    e = get_2d_ellipse_mplpatch(
        qarand.reshape(2, 1), qbrand.reshape(2, 1), cMax=1.3 * lrand, cMin=lrand
    )
    # ax.add_patch(e)
    # ax.plot(qarand[0], qarand[1], "ro", label="q_a" if i == 0 else "")
    # ax.plot(qbrand[0], qbrand[1], "go", label="q_b" if i == 0 else "")
for i in range(n):
    qcenter = Qcircle[i, :]
    c = get_circle_mplpatch(qcenter, roffset=0.5)
    ax.add_patch(c)
    ax.plot(qcenter[0], qcenter[1], "bo", label="q_center" if i == 0 else "")
ax.set_aspect("equal", "box")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.grid(True)
ax.legend()
plt.show()
