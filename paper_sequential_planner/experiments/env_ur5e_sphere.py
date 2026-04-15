import os
import tqdm
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from eaik.IK_DH import DhRobot
from pytransform3d.transform_manager import TransformManager
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class RobotUR5eKin:

    def __init__(self):
        self.d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        self.alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
        self.a = np.array([0, -0.425, -0.3922, 0, 0, 0])

        self.bot = DhRobot(self.alpha, self.a, self.d)

    def solve_fk(self, q):
        return self.bot.fwdKin(q)

    def solve_aik(self, H):
        sols = self.bot.IK(H)
        numsols = sols.num_solutions()
        Q = sols.Q
        return numsols, Q

    @staticmethod
    def _dh_transform(a, alpha, d, theta):
        """Standard DH homogeneous transform from frame i-1 to frame i."""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def get_dh_chain(self, q):
        """Return per-link DH transforms A_i and cumulative base transforms T_0i."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,):
            raise ValueError("q must be a 6-element joint vector for UR5e")

        relative_tf = []
        world_tf = [np.eye(4)]
        T = np.eye(4)

        for i in range(6):
            A_i = self._dh_transform(self.a[i], self.alpha[i], self.d[i], q[i])
            relative_tf.append(A_i)
            T = T @ A_i
            world_tf.append(T.copy())

        return relative_tf, world_tf

    def plot_link_transforms(self, q, frame_size=0.08):
        """Plot all frame transforms derived from DH parameters for a joint state."""
        relative_tf, world_tf = self.get_dh_chain(q)

        tm = TransformManager()
        for i, A_i in enumerate(relative_tf, start=1):
            tm.add_transform(f"link_{i}", f"link_{i-1}", A_i)

        ax = make_3d_axis(ax_s=1.0)

        # Plot frame axes for base and each link frame.
        for i, T_0i in enumerate(world_tf):
            plot_transform(ax=ax, A2B=T_0i, s=frame_size, name=f"L{i}")

        # Draw link-centerline segments between frame origins.
        origins = np.array([T_0i[:3, 3] for T_0i in world_tf])
        ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], "k-o", linewidth=2)

        reach = np.sum(np.abs(self.a)) + np.sum(np.abs(self.d))
        lim = max(0.8, 1.1 * reach)
        # ax.set_xlim([-lim, lim])
        # ax.set_ylim([-lim, lim])
        # ax.set_zlim([0.0, lim])
        # ax.set_xlabel("X [m]")
        # ax.set_ylabel("Y [m]")
        # ax.set_zlabel("Z [m]")
        # ax.set_title("UR5e DH Link Transformations")
        # ax.set_box_aspect([1, 1, 1])

        # A_chain, T_chain = robot_kin.plot_link_transforms(q)
        # for i, A_i in enumerate(A_chain, start=1):
        #     print(f"A_{i} (link_{i-1} -> link_{i}):\n", A_i)
        # for i, T_0i in enumerate(T_chain):
        #     print(f"T_0{i} (base -> link_{i}):\n", T_0i)

        # return relative_tf, world_tf

        return ax


if __name__ == "__main__":
    robot_kin = RobotUR5eKin()
    q = np.array([0, 0, 0, 0, 0, 0])

    FKeaik = robot_kin.solve_fk(q)
    print("FK (end-effector pose):\n", FKeaik)

    ax1 = robot_kin.plot_link_transforms(q)
    plot_transform(ax=ax1, A2B=np.eye(4), s=1.5, name="WORLD")

    with open(
        os.path.join(rsrc, "./ur5e/ur5e_extract_calibrated_spherized.urdf"), "r"
    ) as f:
        URDF = f.read()

    with open(os.path.join(rsrc, "./ur5e/shelf.urdf"), "r") as f:
        URDF_shelf = f.read()

    tm = UrdfTransformManager()
    tm.load_urdf(URDF)
    tm.set_joint("shoulder_pan_joint", q[0])
    tm.set_joint("shoulder_lift_joint", q[1])
    tm.set_joint("elbow_joint", q[2])
    tm.set_joint("wrist_1_joint", q[3])
    tm.set_joint("wrist_2_joint", q[4])
    tm.set_joint("wrist_3_joint", q[5])
    FKurdf = tm.get_transform("tool0", "base_link")
    print(f"==>> FKurdf: \n{FKurdf}")

    position = [0, 0.75, 0]
    quat = [0, 0, 1, 0]
    A2B = np.eye(4)
    A2B[:3, :3] = R.from_quat(quat).as_matrix()
    A2B[:3, 3] = np.asarray(position, dtype=float)
    tm.load_urdf(URDF_shelf)
    tm.add_transform("base", "base_link", A2B)
    tm.plot_collision_objects("base_link")
    tm.plot_visuals("base_link")

    pos2 = [0, -0.75, 0]
    quat2 = [0, 0, 0, 1]
    A2B2 = np.eye(4)
    A2B2[:3, :3] = R.from_quat(quat2).as_matrix()
    A2B2[:3, 3] = np.asarray(pos2, dtype=float)
    URDF_shelf_1 = URDF_shelf.replace("base", "base_1")
    tm.load_urdf(URDF_shelf_1)
    tm.add_transform("base_1", "base_link", A2B2)
    tm.plot_collision_objects("base_link")
    tm.plot_visuals("base_link")

    pos3 = [0.75, 0, 0]
    quat3 = [0, 0, 0.5, 0.5]
    A2B3 = np.eye(4)
    A2B3[:3, :3] = R.from_quat(quat3).as_matrix()
    A2B3[:3, 3] = np.asarray(pos3, dtype=float)
    URDF_shelf_2 = URDF_shelf.replace("base", "base_2")
    tm.load_urdf(URDF_shelf_2)
    tm.add_transform("base_2", "base_link", A2B3)
    tm.plot_collision_objects("base_link")
    tm.plot_visuals("base_link")

    plt.show()
