import numpy as np
import matplotlib.pyplot as plt

try:
    from eaik.IK_DH import DhRobot
    from spatialmath import SE3
    from roboticstoolbox import DHRobot, RevoluteDH
except:
    print("missing packages; usage limited")

np.set_printoptions(precision=3, suppress=True, linewidth=200)


class RobotUR5eKin:

    def __init__(self):
        self.d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        self.alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
        self.a = np.array([0, -0.425, -0.3922, 0, 0, 0])
        self.qlim = np.array(
            [
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-np.pi, np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
            ]
        )
        Ls = [
            RevoluteDH(
                d=self.d[i], a=self.a[i], alpha=self.alpha[i], qlim=self.qlim[i]
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


if __name__ == "__main__":
    robot = RobotUR5eKin()

    q = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
    H = robot.solve_fk(q)
    print("FK H:\n", H)

    M = robot.solve_manipulability(q)
    print("manipulability:", M)

    Jac = robot.solve_jacobian(q)
    print("Jacobian:\n", Jac)
