import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.utils import Utils
from scipy.spatial.transform import Rotation as R

try:
    from eaik.IK_DH import DhRobot
    from spatialmath import SE3
    from roboticstoolbox import DHRobot, RevoluteDH
except:
    print("missing packages; usage limited")

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)


class RobotUR5eKin:

    def __init__(self):
        self.d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        self.alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
        self.a = np.array([0, -0.425, -0.3922, 0, 0, 0])
        self.qlim_dict = {
            "hardware_default": np.array(
                [
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-np.pi, np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                ]
            ),
            "table_under": np.array(
                [
                    [-2 * np.pi, 2 * np.pi],
                    [-np.pi, np.pi],
                    [-np.pi, np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                ]
            ),
        }

        Ls = [
            RevoluteDH(
                d=self.d[i],
                a=self.a[i],
                alpha=self.alpha[i],
                qlim=self.qlim_dict["hardware_default"][i],
            )
            for i in range(6)
        ]

        # bot kinematics
        self.bot = DhRobot(self.alpha, self.a, self.d)
        self.bot_rtb = DHRobot(Ls, name="UR5e")

    def solve_fk(self, q):
        return self.bot.fwdKin(q)

    def solve_aik(self, H):
        sols = self.bot.IK(H)
        numsols = sols.num_solutions()
        Q = sols.Q
        return numsols, Q

    def solve_nik(self, H, q0):
        sol = self.bot_rtb.ikine_LM(H, q0=q0)
        return sol.q

    def solve_aik_bulk(self, H):
        num_sols = []
        ik_sols = []
        for h in H:
            num_sol, ik_sol = self.solve_aik(h)
            num_sols.append(num_sol)
            ik_sols.append(ik_sol)
        return num_sols, ik_sols

    def solve_aik_altconfig(self, H):
        numsol, Qik = self.solve_aik(H)
        limt6 = self.qlim_dict["hardware_default"]
        ikaltconfig = []
        for i in range(numsol):
            alt = Utils.find_alt_config(Qik[i].reshape(6, 1), limt6)
            ikaltconfig.append(alt)
        ikaltconfig = np.hstack(ikaltconfig).T
        return ikaltconfig.shape[0], ikaltconfig

    def solve_aik_altconfig_bulk(self, H):
        num_sols = []
        ik_sols = []
        for h in H:
            num_sol, ik_solaltconfig = self.solve_aik_altconfig(h)
            num_sols.append(num_sol)
            ik_sols.append(ik_solaltconfig)
        return num_sols, ik_sols

    def solve_manipulability(self, q):
        return self.bot_rtb.manipulability(q)

    def solve_jacobian(self, q):
        return self.bot_rtb.jacob0(q)

    def _convert_urdf_to_dh_frame(H):
        "from our design task in urdf frame to dh frame"
        Hdh_to_urdf = SE3.Rz(np.pi).A
        return np.linalg.inv(Hdh_to_urdf) @ H

    def _convert_dh_to_urdf_frame(H):
        "from dh frame to our design task in urdf frame"
        Hdh_to_urdf = SE3.Rz(np.pi).A
        return Hdh_to_urdf @ H


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


def generate_random_dh_tasks(bot, num_tasks=10):
    angle = np.random.uniform(-np.pi, np.pi, size=(num_tasks, 6))
    T = []
    for i in range(num_tasks):
        t = solve_fk(bot, angle[i])
        T.append(t)
    return T


def generate_random_task_transformation():
    translation = np.random.uniform(-1, 1, size=(3,))
    rotation = np.random.uniform(-np.pi, np.pi, size=(4,))
    transformation = np.eye(4)
    RR = R.from_quat(rotation)
    transformation[:3, :3] = RR.as_matrix()
    transformation[:3, 3] = translation
    return transformation


def generate_linear_tasks_transformation(
    s=[1, 1, 1],
    e=[1, -1, 1],
    quat=[0.0, 0.707106, 0.0, 0.707106],
    num_tasks=10,
):
    t = np.linspace(s, e, num_tasks)
    Hlist = [np.eye(4) for _ in range(num_tasks)]
    for i in range(num_tasks):
        Hlist[i][:3, 3] = t[i]
        Hlist[i][:3, :3] = R.from_quat(quat).as_matrix()
    return Hlist


def generate_linear_grid_tasks_transformation():
    size = 4
    H1 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.6], [0.5, -0.5, 0.6], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H2 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.4], [0.5, -0.5, 0.4], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H3 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.2], [0.5, -0.5, 0.2], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H4 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.707106, 0.0, 0.707106], size
    )
    Hlist = H1 + H2 + H3 + H4
    for i in range(len(Hlist)):
        Hlist[i] = convert_urdf_to_dh_frame(Hlist[i])
    return Hlist


def generate_linear_dual_side_tasks_transformation():
    size = 4
    H1 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.6], [0.5, -0.5, 0.6], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H2 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.4], [0.5, -0.5, 0.4], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H3 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.2], [0.5, -0.5, 0.2], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H4 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.707106, 0.0, 0.707106], size
    )
    Hlist = H1 + H2 + H3 + H4
    H1list = []
    for h in Hlist:
        H1list.append(convert_urdf_to_dh_frame(h))
    return Hlist + H1list


def generate_spiral_task_transformation():
    turns = 5  # number of full rotations
    points_per_turn = 5  # resolution along the curve
    height = 0.7  # total height of the spiral
    radius = 0.5  # constant radius (for a helix). Change below for conical

    # Parametric variable
    t = np.linspace(0, 2 * np.pi * turns, points_per_turn * turns)

    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = (height / (2 * np.pi * turns)) * t

    H = []
    for i in range(len(x)):
        Hi = np.eye(4)
        Hi[:3, 3] = np.array([x[i], y[i], z[i]])
        Hi[:3, :3] = R.from_quat([0.0, 0.707106, 0.0, 0.707106]).as_matrix()
        H.append(Hi)

    basisvector = np.array([1, 0])
    Hx = []
    for i in range(len(x)):
        xyvector = np.array([x[i], y[i]])
        alpha = np.arccos(np.dot(basisvector, xyvector) / np.linalg.norm(xyvector))
        Hx.append(SE3.Rx(alpha).A)

    for i in range(len(H)):
        H[i] = H[i] @ Hx[i]
    return H


if __name__ == "__main__":
    robot = RobotUR5eKin()

    q = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
    H = robot.solve_fk(q)
    print("FK H:\n", H)

    M = robot.solve_manipulability(q)
    print("manipulability:", M)

    Jac = robot.solve_jacobian(q)
    print("Jacobian:\n", Jac)
