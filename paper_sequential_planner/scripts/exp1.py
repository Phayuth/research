# follow the taskspace tour via numerical inverse kinematics
# this is more of a control problem
import numpy as np
from spatialmath import SE3
import util


bot = util.ur5e_dh()
botrtb = util.ur5e_rtb_dh()


def task_to_task_interpolation(task1, task2, step):
    if not isinstance(task1, SE3):
        task1 = SE3(task1)
        task2 = SE3(task2)
    Hinterp = [task1.interp(task2, t) for t in np.linspace(0, 1, step)]
    return Hinterp


def check_joint_limit(q, qlimit):
    ck = []
    for i in range(len(q)):
        if q[i] < qlimit[i][0] or q[i] > qlimit[i][1]:
            ck.append(False)
        else:
            ck.append(True)
    return all(ck)


def move_in_taskspace(H2, qinit):
    """move to H2  using taskspace interpolation with jacobian inverse
    xdot = J * qdot
    qdot = J_inv * xdot
    """
    ur5ertb = util.ur5e_rtb_dh()
    qik = ur5ertb.ikine_NR(H2, q0=qinit)
    return qik


def check_tspace_move_over_jointlimit(qinit, H1, H2, qlimit):
    """
    move from H1 to H2 using taskspace interpolation with jacobian inverse
    it is a locally optimal path and if we start from qinit, we might hit limit.
    """
    qinitog = qinit.copy()
    Hinterp = task_to_task_interpolation(H1, H2, 10)
    for i in range(1, len(Hinterp)):
        H = Hinterp[i]
        qsol, success = util.ikine_min_norm(util.ur5e_dh(), H, qinit)
        if not success:
            print("ikine failed")
            return False, qinitog
        if not check_joint_limit(qsol, qlimit):
            print("hit joint limit")
            return False, qinitog
        qinit = qsol


if __name__ == "__main__":
    bot = util.ur5e_dh()
    H = util.generate_random_dh_tasks(bot, 10)
    s1 = H[0]
    s2 = H[1]
    steps = 5
    traj = task_to_task_interpolation(s1, s2, steps)
    for i, t in enumerate(traj):
        print(f"step {i}: {t}")

    q = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
    qlimit = [(-2 * np.pi, 2 * np.pi)] * 6
    qlimit[2] = (-np.pi, np.pi)
    print(qlimit)
    print("check_joint_limit", check_joint_limit(q, qlimit))
