import numpy as np
import time
import util
from spatialmath.base import qslerp
from spatial_geometry.utils import Utils


def position_euclidean_distance(p1, p2):
    return np.linalg.norm(p2 - p1)


def orientation_quaternion_distance(quat1, quat2):
    dot_product = np.abs(np.dot(quat1, quat2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = 2 * np.arccos(dot_product)
    return angle


def taskspace_distance(t1, t2, w_pos=1.0, w_ori=1.0):
    pos1, ori1 = t1
    pos2, ori2 = t2
    pos_dist = position_euclidean_distance(pos1, pos2)
    ori_dist = orientation_quaternion_distance(ori1, ori2)
    return w_pos * pos_dist + w_ori * ori_dist


def configspace_euclidean_distance(q1, q2):
    return np.linalg.norm(q2 - q1)


def configspace_weighted_euclidean_distance(q1, q2, weights):
    """
    give weights for each joint
    `"""
    weighted_diff = weights * (q2 - q1)
    return np.linalg.norm(weighted_diff)


def configspace_max_euclidean_distance(q1, q2):
    return np.max(np.abs(q2 - q1))


def wraptopi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def configspace_torus_distance(q1, q2):
    q1 = wraptopi(q1)
    q2 = wraptopi(q2)
    delta = np.abs(q1 - q2)
    delta = np.where(delta > np.pi, 2 * np.pi - delta, delta)
    return np.linalg.norm(delta)


def check_task_move_over_bound(qinit, task1, task2, qlimit):
    """
    move from task1 to task2 using taskspace interpolation with jacobian inverse
    it is a locally optimal path and if we start from qinit, we might hit limit.
    """


def task_to_task_interpolation(task1, task2, dt=0.1):
    pos1, quat1 = task1
    pos2, quat2 = task2

    # fake
    return taskinterp


def move_in_taskspace(tasks, taskg, qinit):
    """
    move to taskg using taskspace interpolation with jacobian inverse
    xdot = J(q) * qdot
    qdot = J_inv * xdot
    """


bot = util.ur5e_dh()

T = util.generate_random_dh_tasks(bot, 2)
T1 = T[0]
T2 = T[1]


n1, Q1 = util.solve_ik(bot, T1)
n2, Q2 = util.solve_ik(bot, T2)
# the solution is always in the range of -pi to pi

print("T1", T1)
print("Q1", Q1)
print("T2", T2)
print("Q2", Q2)

diffQ = Q2 - Q1
length = np.linalg.norm(diffQ, axis=1)
print("length", length)

length_torus = np.zeros(n1)
for i in range(n1):
    q1 = Q1[i]
    q2 = Q2[i]
    dd = configspace_torus_distance(q1, q2)
    length_torus[i] = dd
print("length_torus", length_torus)

l = np.zeros((n1, n2))
for i in range(n1):
    for j in range(n2):
        q1 = Q1[i]
        q2 = Q2[j]
        diffq = q2 - q1
        length = np.linalg.norm(diffq)
        # print(f"i={i} j={j} length={length:.4f}")
        l[i, j] = length
print("l\n", l)

lsort = np.sort(l, axis=None)
print("lsort", lsort)


le = np.zeros((n1, n2))
for i in range(n1):
    for j in range(n2):
        q1 = Q1[i]
        q2 = Q2[j]
        dd = configspace_torus_distance(q1, q2)
        le[i, j] = dd
print("le\n", le)

lesort = np.sort(le, axis=None)
print("lesort", lesort)
