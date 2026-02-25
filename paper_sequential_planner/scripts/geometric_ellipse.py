import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


# informed sampling according to the paper
def informed_sampling(xStart, xGoal, cMax):
    xCenter = (xStart + xGoal) / 2
    rotationAxisC = rotation_to_world(xStart, xGoal)
    cMin = np.linalg.norm(xGoal - xStart)
    L = hyperellipsoid_informed_axis_length(cMax, cMin)
    xBall = unit_ball_sampling()
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand


def informed_sampling_bulk(xStart, xGoal, cMax, numsample):
    xCenter = (xStart + xGoal) / 2
    rotationAxisC = rotation_to_world(xStart, xGoal)
    cMin = np.linalg.norm(xGoal - xStart)
    L = hyperellipsoid_informed_axis_length(cMax, cMin)
    xBall = unit_ball_sampling_bulk(xStart.shape[0], numsample)
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand.T


def informed_surface_sampling_bulk(xStart, xGoal, cMax, numsample):
    xCenter = (xStart + xGoal) / 2
    rotationAxisC = rotation_to_world(xStart, xGoal)
    cMin = np.linalg.norm(xGoal - xStart)
    L = hyperellipsoid_informed_axis_length(cMax, cMin)
    xBall = unit_ball_surface_sampling_bulk(xStart.shape[0], numsample)
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand.T


def hyperellipsoid_informed_axis_length(cMax, cMin, dof=2):  # L
    r1 = cMax / 2
    ri = np.sqrt(cMax**2 - cMin**2) / 2
    diagTerm = [r1] + [ri] * (dof - 1)
    return np.diag(diagTerm)


# custom sampling on ellipse with custom axis lengths
def elliptical_sampling(xStart, xGoal, L, sampling):
    xCenter = (xStart + xGoal) / 2
    rotationAxisC = rotation_to_world(xStart, xGoal)
    xBall = sampling()
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand


def hyperellipsoid_custom_axis_length(long_axis, short_axis, dof=2):  # L
    diagTerm = [long_axis] + [short_axis] * (dof - 1)
    return np.diag(diagTerm)


def custom_inside_sampling(xStart, xGoal, long_axis, short_axis, numsample):
    L = hyperellipsoid_custom_axis_length(long_axis, short_axis)
    xCenter = (xStart + xGoal) / 2
    rotationAxisC = rotation_to_world(xStart, xGoal)
    xBall = unit_ball_sampling_bulk(xStart.shape[0], numsample)
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand.T


def custom_surface_sampling(xStart, xGoal, long_axis, short_axis, numsample):
    L = hyperellipsoid_custom_axis_length(long_axis, short_axis)
    xCenter = (xStart + xGoal) / 2
    rotationAxisC = rotation_to_world(xStart, xGoal)
    xBall = unit_ball_surface_sampling_bulk(xStart.shape[0], numsample)
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand.T


# inside sampling or on the surface sampling
def unit_ball_sampling(dof=2):
    u = np.random.normal(0.0, 1.0, (dof + 2, 1))
    norm = np.linalg.norm(u)
    u = u / norm
    return u[:dof, :]  # The first N coordinates are uniform in a unit N ball


def unit_ball_sampling_bulk(dof=2, num_samples=1):
    u = np.random.normal(0.0, 1.0, (dof + 2, num_samples))
    norms = np.linalg.norm(u, axis=0)
    u = u / norms
    return u[:dof, :]  # The first N coordinates are uniform in a unit N ball


def unit_ball_surface_sampling(dof=2):
    u = np.random.normal(0.0, 1.0, (dof, 1))
    norm = np.linalg.norm(u)
    u = u / norm
    return u  # The first N coordinates are uniform on the surface of a unit N ball


def unit_ball_surface_sampling_bulk(dof=2, num_samples=1):
    u = np.random.normal(0.0, 1.0, (dof, num_samples))
    norms = np.linalg.norm(u, axis=0)
    u = u / norms
    return u  # The first N coordinates are uniform on the surface of a unit N ball


# Rotation
def rotation_to_world(xStart, xGoal):  # C
    dof = xStart.shape[0]
    cMin = np.linalg.norm(xGoal - xStart)
    a1 = (xGoal - xStart) / cMin
    I1 = np.array([1.0] + [0.0] * (dof - 1)).reshape(1, -1)
    M = a1 @ I1
    U, _, V_T = np.linalg.svd(M, True, True)
    middleTerm = [1.0] * (dof - 1) + [np.linalg.det(U) * np.linalg.det(V_T.T)]
    return U @ np.diag(middleTerm) @ V_T


def sampling_rotation_matrix(n):
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    # fix sign ambiguity
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    # enforce det = +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def isPointinEllipse(ecx, ecy, ela, elb, eRM, px, py):
    ellipsoidCenter = np.array([ecx, ecy]).reshape(2, 1)
    ellipsoidAxis = np.array([ela, elb]).reshape(2, 1)
    pointCheck = np.array([px, py]).reshape(2, 1) - ellipsoidCenter
    pointCheckRotateBack = eRM.T @ pointCheck
    mid = pointCheckRotateBack / ellipsoidAxis
    midsq = mid**2
    eq = sum(midsq)
    if eq <= 1.0:
        return True
    else:
        return False

def isPointinEllipseBulk(ecx, ecy, ela, elb, eRM, points):
    ellipsoidCenter = np.array([ecx, ecy]).reshape(2, 1)
    ellipsoidAxis = np.array([ela, elb]).reshape(2, 1)
    pointCheck = points.T - ellipsoidCenter
    pointCheckRotateBack = eRM.T @ pointCheck
    mid = pointCheckRotateBack / ellipsoidAxis
    midsq = mid**2
    eq = np.sum(midsq, axis=0)
    return eq <= 1.0

#
def sampling_circle_in_jointlimit(roffset, dof=2):
    limit = np.array(
        [
            [-np.pi, np.pi],
        ]
        * dof
    )
    limit_offset = limit.copy()
    limit_offset[:, 0] += roffset
    limit_offset[:, 1] -= roffset
    qcenter = np.random.uniform(
        low=limit_offset[:, 0], high=limit_offset[:, 1], size=(dof,)
    )
    return qcenter


#
def sampling_two_points(eta, dof=2):
    qa = np.random.uniform(low=-np.pi, high=np.pi, size=(dof,))
    qb = np.random.uniform(low=-np.pi, high=np.pi, size=(dof,))
    qc = qa + eta * (qa - qb) / np.linalg.norm(qa - qb)
    l = np.linalg.norm(qa - qc)
    return qa, qc, l


#
def sampling_Xstartgoal(qcenter, Rbest, cmin, dof=2):
    if cmin > 2 * Rbest:
        raise ValueError("cmin is larger than the diameter of the circle.")

    Rrand = sampling_rotation_matrix(dof)
    cMax = 2 * Rbest
    L = hyperellipsoid_informed_axis_length(cMax, cmin, dof=dof)
    e1 = np.zeros((dof, 1))
    e1[0, 0] = 1.0
    u = Rrand @ e1
    rmin = cmin / 2
    qa = qcenter - rmin * u
    qb = qcenter + rmin * u
    return qa, qb


# util function
def three_point_path(qs, qm, qe, num_points=100):
    # intended for interpolate between qs, qmid on ellipse and qe
    p1 = np.linspace(qs, qm, num=num_points // 2)
    p2 = np.linspace(qm, qe, num=num_points // 2)
    p = np.vstack((p1, p2))
    return p[:, :, 0].T  # shape (num_points, dof)


def linear_interp(qa, qb, eta=0.1):
    dist = np.linalg.norm(qb - qa)
    num_segments = int(np.ceil(dist / eta))
    path = []
    for i in range(num_segments + 1):
        alpha = i / num_segments
        q = (1 - alpha) * qa + alpha * qb
        path.append(q)
    path = np.array(path)
    return path


def bezier_curve(q0, qc, qg, num_points=100):
    t = np.linspace(0, 1, num_points)
    B = ((1 - t) ** 2) * q0 + (2 * (1 - t) * t) * qc + (t**2) * qg
    return B


#
# The Lebesgue measure (i.e., "volume") of an n-dimensional ball with a unit radius.
def unit_nball_volume_measure(dof):
    gammafunction = {
        1: 1.0,
        2: 1.0,
        3: 2.0,
        4: 6.0,
        5: 24.0,
        6: 120.0,
    }
    return (np.pi ** (dof / 2)) / gammafunction[(dof / 2) + 1]  # ziD


# The Lebesgue measure (i.e., "volume") of an n-dimensional prolate hyperspheroid
# (a symmetric hyperellipse) given as the distance between the foci and the transverse diameter.
def prolate_hyperspheroid_measure():
    pass


def lebesgue_obstacle_free_measure(configlimit):
    # configLimit = [
    #     [-np.pi, np.pi],
    #     [-np.pi, np.pi],
    #     [-np.pi, np.pi],
    #     [-np.pi, np.pi],
    #     [-np.pi, np.pi],
    #     [-np.pi, np.pi],
    # ]
    diff = np.diff(configlimit)
    return np.prod(diff)


def rewire_radius(eta, dof, configlimit, numVert, rwfact=1.1):
    inverseDoF = 1.0 / dof
    lbf = lebesgue_obstacle_free_measure(configlimit)
    ubvm = unit_nball_volume_measure(dof)
    gammaRRG = rwfact * 2.0 * ((1.0 + inverseDoF) * (lbf / ubvm)) ** (inverseDoF)
    return np.min([eta, gammaRRG * (np.log(numVert) / numVert) ** (inverseDoF)])


# Matplotlib patch for 2D ellipse visualization
def get_2d_ellipse_informed_mplpatch(xStart, xGoal, cMax):
    cMin = np.linalg.norm(xGoal - xStart)
    xCenter = (xStart + xGoal) / 2
    L = hyperellipsoid_informed_axis_length(cMax, cMin)
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


def get_2d_ellipse_custom_mplpatch(xStart, xGoal, long_axis, short_axis):
    xCenter = (xStart + xGoal) / 2
    L = hyperellipsoid_custom_axis_length(long_axis, short_axis)
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


def get_2d_circle_mplpatch(qcenter, r):
    circle = plt.Circle(
        (qcenter[0], qcenter[1]),
        r,
        color="b",
        fill=False,
        linestyle="--",
    )
    return circle


def interplolate_line(q1, q2, n=10):
    path = np.linspace(q1, q2, n)
    return path


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def unit_vector(q1, q2):
    v = q2 - q1
    return v / np.linalg.norm(v)


def sorting_sampling_():
    # xfake = np.linspace(0, 10, 100)
    # yfake = np.sin(xfake)
    # X = np.vstack((xfake, yfake)).T
    # v = np.array([1, 0])

    X = np.array(
        [
            [0.5, 1.5],
            [1.5, 1.0],
            [1.0, 0.5],
            [2.5, 1.0],
            [2.0, 2.5],
            [3.5, 3.0],
            [3.0, 3.5],
        ]
    )
    q1 = np.array([0, 0])
    q2 = np.array([1, 1])
    v = unit_vector(q1, q2)
    print("Projection direction v:", v)

    s = X @ v  # projection scalars
    order = np.argsort(s)
    X_sorted = X[order]

    print("Original X:")
    print(X)
    print("Projection scalars:")
    print(s)
    print("Sorted order indices:")
    print(order)
    print("Sorted X:")
    print(X_sorted)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color="blue", label="Original Points")
    ax.scatter(X_sorted[:, 0], X_sorted[:, 1], color="red", label="Sorted Points")

    for i in range(X.shape[0]):
        ax.text(X[i, 0] + 0.1, X[i, 1], f"{i}", fontsize=12, color="blue")
        ax.text(X_sorted[i, 0], X_sorted[i, 1], f"{i}", fontsize=12, color="red")
    ax.plot(
        [0, v[0] * 6],
        [0, v[1] * 6],
        color="green",
        linestyle="--",
        label="Projection Direction",
    )
    ax.legend()
    ax.grid()
    ax.set_aspect("equal", "box")
    plt.show()


if __name__ == "__main__":
    sorting_sampling_()

    theta = 1.0
    rM = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    ellipse = (0, 4, 5, 3, rM)
    p1 = (4.58, 1.14)
    p2 = (0.0, 0.0)
    p3 = (4.0, 3.8)
    p4 = (-2.5, 6.4)
    points = np.array([p1, p2, p3, p4])
    states = isPointinEllipseBulk(*ellipse, points)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.grid()
    e = patches.Ellipse(
        (ellipse[0], ellipse[1]),
        width=ellipse[2] * 2,
        height=ellipse[3] * 2,
        angle=np.degrees(np.arctan2(ellipse[4][1, 0], ellipse[4][0, 0])),
        fill=False,
        edgecolor="red",
        linewidth=2,
    )
    ax.add_patch(e)
    ax.scatter(p1[0], p1[1], color="red", label=f"p1, state: {states[0]}")
    ax.scatter(p2[0], p2[1], color="blue", label=f"p2, state: {states[1]}")
    ax.scatter(p3[0], p3[1], color="green", label=f"p3, state: {states[2]}")
    ax.scatter(p4[0], p4[1], color="orange", label=f"p4, state: {states[3]}")
    ax.legend()
    plt.show()