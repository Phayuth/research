import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from scipy.spatial.transform import Rotation as Rot
from spatial_geometry.utils import Utils
from matplotlib import collections as mc
from scipy.spatial import ConvexHull
from matplotlib.backend_bases import MouseButton

# init
limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
qinit = np.array([0.0, 0.0])
Qinit = Utils.find_alt_config(qinit.reshape(2, 1), limt2)

# possible goal set
qmean = np.array([4, 2])
Covxy = 0.01 * np.array([[6, 0], [0, 6]])
Qset = Utils.find_alt_config(qmean.reshape(2, 1), limt2)
npts = 1000

# current single goal
qgoal = np.array([0.0, 0.0])
Qgoalset = Utils.find_alt_config(qgoal.reshape(2, 1), limt2)

# cspace collision point
rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
collision = np.load(rsrc + "collisionpoint_exts.npy")

convexhullGoalsetOnly = True


def plot_cov_ellipse(cov, pos, nstd=1, **kwargs):
    # nstd is the number of standard deviations to determine the ellipse
    # nstd = 1 is roughly 68% confidence interval
    # nstd = 2 is roughly 95% confidence interval
    # nstd = 3 is roughly 99.7% confidence interval
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
    return ellipse


# single ellipse plot
if False:
    pts = np.random.multivariate_normal(q, Covxy, size=npts)
    fig, ax = plt.subplots()
    ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, alpha=0.5)
    ellipse = plot_cov_ellipse(Covxy, q, nstd=3, edgecolor="red", fc="None", lw=2)
    ax.add_patch(ellipse)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    fig.tight_layout()
    plt.show()


def make_rot2d(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def is_point_inside_ellipse(pos, width, height, anglerad, point):
    ellipsoidCenter = pos.reshape(2, 1)
    ellipsoidAxis = np.array([width, height]).reshape(2, 1)
    pointCheck = point - ellipsoidCenter
    pointCheckRotateBack = make_rot2d(anglerad) @ pointCheck
    mid = pointCheckRotateBack / ellipsoidAxis
    midsq = mid**2
    eq = sum(midsq)
    if eq <= 1.0:
        return True
    else:
        return False


# pos, width, height, angle = find_ellipse(Covxy, q)
# print(f"> pos: {pos}")
# print(f"> width: {width}")
# print(f"> height: {height}")
# print(f"> angle: {angle}")

# point = np.array([4, 2]).reshape(2, 1)
# is_inside = is_point_inside_ellipse(pos, width, height, np.deg2rad(angle), point)
# print(f"> is_inside: {is_inside}")


def find_ellipse(cov, pos, nstd=1):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    return pos, width, height, angle


def which_point_in_polygon(pointlist, polys):
    for i, point in enumerate(pointlist):
        if polys.contains(point):
            return np.array([point.x, point.y])
        else:
            return None


pts1 = np.random.multivariate_normal(Qset[:, 0], Covxy, size=npts)
pts2 = np.random.multivariate_normal(Qset[:, 1], Covxy, size=npts)
pts3 = np.random.multivariate_normal(Qset[:, 2], Covxy, size=npts)
pts4 = np.random.multivariate_normal(Qset[:, 3], Covxy, size=npts)
ptss = np.vstack([pts1, pts2, pts3, pts4])

if True:
    fig, ax = plt.subplots()

    ax.plot(
        collision[:, 0],
        collision[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )
    ax.plot(pts1[:, 0], pts1[:, 1], ".", markersize=1, alpha=0.5)
    ax.plot(pts2[:, 0], pts2[:, 1], ".", markersize=1, alpha=0.5)
    ax.plot(pts3[:, 0], pts3[:, 1], ".", markersize=1, alpha=0.5)
    ax.plot(pts4[:, 0], pts4[:, 1], ".", markersize=1, alpha=0.5)

    ellipselist = [
        plot_cov_ellipse(Covxy, qi, nstd=3, edgecolor="red", fc="None", lw=2)
        for qi in Qset.T
    ]
    for el in ellipselist:
        ax.add_patch(el)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(0, color="k", linestyle="--")
    ax.axvline(-np.pi, color="m", linestyle="--")
    ax.axhline(np.pi, color="c", linestyle="--")
    ax.axvline(np.pi, color="m", linestyle="--")
    ax.axhline(-np.pi, color="c", linestyle="--")
    ax.axvline(-2 * np.pi, color="m", linestyle="-")
    ax.axhline(2 * np.pi, color="c", linestyle="-")
    ax.axvline(2 * np.pi, color="m", linestyle="-")
    ax.axhline(-2 * np.pi, color="c", linestyle="-")
    ax.set_xlim(-2 * np.pi - 0.1, 2 * np.pi + 0.1)
    ax.set_ylim(-2 * np.pi - 0.1, 2 * np.pi + 0.1)
    ax.set_aspect("equal")
    ax.set_title(f"Cspace Full View")

    (lineinit,) = ax.plot([], [], "x-", markersize=10, color="blue")

    (linexy1,) = ax.plot([], [], "r-", lw=2)
    (linexy2,) = ax.plot([], [], "r-", lw=2)
    (linexy3,) = ax.plot([], [], "r-", lw=2)
    (linexy4,) = ax.plot([], [], "r-", lw=2)

    (linerect,) = ax.plot([], [], "o-", lw=2)

    d = 2
    fig2, axs2 = plt.subplots(d, 1, sharex=False)
    for i in range(d):
        axs2[i].plot([1] * pts1.shape[0], pts1[:, i], ".")
        axs2[i].plot([1] * pts2.shape[0], pts2[:, i], ".")
        axs2[i].plot([1] * pts3.shape[0], pts3[:, i], ".")
        axs2[i].plot([1] * pts4.shape[0], pts4[:, i], ".")
        axs2[i].axhline(-np.pi, color="m", linestyle="--")
        axs2[i].axhline(np.pi, color="c", linestyle="--")
        axs2[i].axhline(
            -2 * np.pi, color="m", linestyle="-", label=f"J Min : -2$\\pi$"
        )
        axs2[i].axhline(
            2 * np.pi, color="c", linestyle="-", label=f"J Max : 2$\\pi$"
        )
        axs2[i].axhline(0, color="k", linestyle="--")
        axs2[i].set_ylabel(f"theta {i+1}")
        axs2[i].set_xlim(0, 1.01)
        axs2[i].legend(bbox_to_anchor=(1, 1), loc="upper left")
        axs2[i].grid(True)
    axs2[-1].set_xlabel("Time")
    axs2[0].set_title("Joint v Time Space View")

    (line1_q1,) = axs2[0].plot([], [], "r-", lw=2)
    (line1_q2,) = axs2[1].plot([], [], "r-", lw=2)
    (line2_q1,) = axs2[0].plot([], [], "r-", lw=2)
    (line2_q2,) = axs2[1].plot([], [], "r-", lw=2)
    (line3_q1,) = axs2[0].plot([], [], "r-", lw=2)
    (line3_q2,) = axs2[1].plot([], [], "r-", lw=2)
    (line4_q1,) = axs2[0].plot([], [], "r-", lw=2)
    (line4_q2,) = axs2[1].plot([], [], "r-", lw=2)

    fig3, ax3 = plt.subplots()

    ax3.plot(
        collision[:, 0],
        collision[:, 1],
        color="darkcyan",
        linewidth=0,
        marker="o",
        markerfacecolor="darkcyan",
        markersize=1.5,
    )
    ax3.plot(pts1[:, 0], pts1[:, 1], ".", markersize=1, alpha=0.5)
    ax3.plot(pts2[:, 0], pts2[:, 1], ".", markersize=1, alpha=0.5)
    ax3.plot(pts3[:, 0], pts3[:, 1], ".", markersize=1, alpha=0.5)
    ax3.plot(pts4[:, 0], pts4[:, 1], ".", markersize=1, alpha=0.5)

    ellipselist = [
        plot_cov_ellipse(Covxy, qi, nstd=3, edgecolor="red", fc="None", lw=2)
        for qi in Qset.T
    ]
    for el in ellipselist:
        ax3.add_patch(el)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.axhline(0, color="k", linestyle="--")
    ax3.axvline(0, color="k", linestyle="--")
    ax3.axvline(-np.pi, color="m", linestyle="--")
    ax3.axhline(np.pi, color="c", linestyle="--")
    ax3.axvline(np.pi, color="m", linestyle="--")
    ax3.axhline(-np.pi, color="c", linestyle="--")
    ax3.axvline(-2 * np.pi, color="m", linestyle="-")
    ax3.axhline(2 * np.pi, color="c", linestyle="-")
    ax3.axvline(2 * np.pi, color="m", linestyle="-")
    ax3.axhline(-2 * np.pi, color="c", linestyle="-")
    ax3.set_xlim(-2 * np.pi - 0.1, 2 * np.pi + 0.1)
    ax3.set_ylim(-2 * np.pi - 0.1, 2 * np.pi + 0.1)
    ax3.set_aspect("equal")
    ax3.set_title("Cspace Scoped View")

    (lineinitscope,) = ax3.plot([], [], "x", markersize=10, color="blue")

    (linexy1scope,) = ax3.plot([], [], "r-", lw=2)
    (linexy2scope,) = ax3.plot([], [], "r-", lw=2)
    (linexy3scope,) = ax3.plot([], [], "r-", lw=2)
    (linexy4scope,) = ax3.plot([], [], "r-", lw=2)

    t = np.linspace(0, 1, 100)

    def click(event):
        if event.xdata is None and event.ydata is None:
            return

        if event.button is MouseButton.RIGHT:
            qinit[0] = event.xdata
            qinit[1] = event.ydata
            global Qinit
            Qinit = Utils.find_alt_config(qinit.reshape(2, 1), limt2)
            QinitsetT = Qinit.T
            orderinitQ = ConvexHull(QinitsetT)

            x1init = np.hstack(
                [
                    QinitsetT[orderinitQ.vertices, 0],
                    QinitsetT[orderinitQ.vertices, 0][0],
                ]
            )
            y1init = np.hstack(
                [
                    QinitsetT[orderinitQ.vertices, 1],
                    QinitsetT[orderinitQ.vertices, 1][0],
                ]
            )
            lineinit.set_data(x1init, y1init)

            lineinitscope.set_data(Qinit[0], Qinit[1])

        if event.button is MouseButton.LEFT:
            qgoal[0] = event.xdata
            qgoal[1] = event.ydata
            Qgoalset = Utils.find_alt_config(qgoal.reshape(2, 1), limt2)
            QgoalsetT = Qgoalset.T
            orderQ = ConvexHull(QgoalsetT)

            x1 = np.hstack(
                [QgoalsetT[orderQ.vertices, 0], QgoalsetT[orderQ.vertices, 0][0]]
            )
            y1 = np.hstack(
                [QgoalsetT[orderQ.vertices, 1], QgoalsetT[orderQ.vertices, 1][0]]
            )
            linerect.set_data(x1, y1)

            line1 = np.linspace(qinit, Qgoalset[:, 0], 100).reshape(-1, 2)
            line2 = np.linspace(qinit, Qgoalset[:, 1], 100).reshape(-1, 2)
            line3 = np.linspace(qinit, Qgoalset[:, 2], 100).reshape(-1, 2)
            line4 = np.linspace(qinit, Qgoalset[:, 3], 100).reshape(-1, 2)

            linexy1.set_data(line1[:, 0], line1[:, 1])
            linexy2.set_data(line2[:, 0], line2[:, 1])
            linexy3.set_data(line3[:, 0], line3[:, 1])
            linexy4.set_data(line4[:, 0], line4[:, 1])

            ax.set_title(
                f"Cspace Full View Clicked at: ({event.xdata:.2f}, {event.ydata:.2f})"
            )

            line1_q1.set_data(t, line1[:, 0])
            line1_q2.set_data(t, line1[:, 1])
            line2_q1.set_data(t, line2[:, 0])
            line2_q2.set_data(t, line2[:, 1])
            line3_q1.set_data(t, line3[:, 0])
            line3_q2.set_data(t, line3[:, 1])
            line4_q1.set_data(t, line4[:, 0])
            line4_q2.set_data(t, line4[:, 1])

            if convexhullGoalsetOnly:
                # Qgoal convexhull only
                xlim = [np.min(QgoalsetT[:, 0]), np.max(QgoalsetT[:, 0])]
                ylim = [np.min(QgoalsetT[:, 1]), np.max(QgoalsetT[:, 1])]
                ax3.set_xlim(xlim)
                ax3.set_ylim(ylim)
            else:
                # Qinit and Qgoal convexhull
                allset = np.vstack([Qinit.T, QgoalsetT])
                xlim = [np.min(allset[:, 0]), np.max(allset[:, 0])]
                ylim = [np.min(allset[:, 1]), np.max(allset[:, 1])]
                ax3.set_xlim(xlim)
                ax3.set_ylim(ylim)

        fig.canvas.draw()
        fig2.canvas.draw()
        fig3.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", click)
    plt.show()
