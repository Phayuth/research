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
def three_point_path(qs, qe, qm):
    # intended for interpolate between qs, qmid on ellipse and qe
    p1 = np.linspace(qs, qm, num=10)
    p2 = np.linspace(qm, qe, num=10)
    p = np.vstack((p1, p2))
    return p


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
