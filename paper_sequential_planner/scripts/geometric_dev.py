import numpy as np


def informed_sampling(xCenter, cMax, cMin, rotationAxisC):
    L = hyperellipsoid_axis_length(cMax, cMin)
    xBall = unit_ball_sampling()
    xRand = (rotationAxisC @ L @ xBall) + xCenter
    return xRand


def hyperellipsoid_axis_length(cMax, cMin):  # L
    r1 = cMax / 2
    ri = np.sqrt(cMax**2 - cMin**2) / 2
    diagTerm = [r1] + [ri] * (dof - 1)
    return np.diag(diagTerm)


def unit_ball_sampling():
    u = np.random.normal(0.0, 1.0, (dof + 2, 1))
    norm = np.linalg.norm(u)
    u = u / norm
    return u[:dof, :]  # The first N coordinates are uniform in a unit N ball


def rotation_to_world(xStart, xGoal):  # C
    cMin = distance_between_config(xStart, xGoal)
    a1 = (xGoal - xStart) / cMin
    I1 = np.array([1.0] + [0.0] * (dof - 1)).reshape(1, -1)
    M = a1 @ I1
    U, _, V_T = np.linalg.svd(M, True, True)
    middleTerm = [1.0] * (dof - 1) + [np.linalg.det(U) * np.linalg.det(V_T.T)]
    return U @ np.diag(middleTerm) @ V_T


def distance_between_config(xFrom, xTo):
    return np.linalg.norm(xTo - xFrom)


dof = 6  # degrees of freedom for the configuration space
xStart = np.array([0.0] * dof).reshape(-1, 1)
xApp = np.array([1.0] * dof).reshape(-1, 1)
xCenter = (xStart + xApp) / 2
C = rotation_to_world(xStart, xApp)  # hyperellipsoid rotation axis
cMin = distance_between_config(xStart, xApp)
cMax = 3.0
xRand = informed_sampling(xCenter, cMax, cMin, C)

print(xRand)
