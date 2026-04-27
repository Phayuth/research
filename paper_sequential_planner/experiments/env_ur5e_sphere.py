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
from pytransform3d.plot_utils import plot_sphere, plot_box
import torch

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
        print(f"==>> relative_tf: \n{relative_tf}")
        print(f"==>> world_tf: \n{world_tf}")

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


def build_sphere_centers_from_fk(ret, collision_sphere, device):
    """Transform link-local collision spheres to base frame for batched FK.

    Returns:
        center_radius_by_link: dict[link_name] -> [num_ik, num_spheres_on_link, 4]
        center_radius_by_sphere: dict["link_name[i]"] -> [num_ik, 4]
        center_radius_all: [num_ik, total_num_spheres, 4]
    """
    center_radius_by_link = {}
    center_radius_by_sphere = {}
    center_radius_all_links = []

    first_link_name = next(iter(collision_sphere))
    first_tf = ret[first_link_name].get_matrix()
    num_ik = first_tf.shape[0]

    for link_name, spheres in collision_sphere.items():
        link_tf = ret[link_name].get_matrix()  # [num_ik, 4, 4]

        centers_local = torch.tensor(
            [s["center"] for s in spheres], dtype=link_tf.dtype, device=device
        )  # [num_spheres, 3]
        centers_hom = torch.cat(
            [
                centers_local,
                torch.ones(
                    centers_local.shape[0], 1, dtype=link_tf.dtype, device=device
                ),
            ],
            dim=1,
        )  # [num_spheres, 4]

        centers_base = torch.einsum("bij,nj->bni", link_tf, centers_hom)[..., :3]

        radii = torch.empty(
            (num_ik, len(spheres)), dtype=link_tf.dtype, device=device
        )
        for si, s in enumerate(spheres):
            # Supports fixed radius or optional per-IK radii in s["radius_by_ik"].
            if "radius_by_ik" in s:
                r = torch.as_tensor(
                    s["radius_by_ik"], dtype=link_tf.dtype, device=device
                )
                if r.ndim == 0:
                    radii[:, si] = r
                elif r.numel() == num_ik:
                    radii[:, si] = r.reshape(num_ik)
                else:
                    raise ValueError(
                        f"radius_by_ik for {link_name}[{si}] must have length {num_ik}, got {r.numel()}"
                    )
            else:
                radii[:, si] = float(s["radius"])

        center_radius_link = torch.cat([centers_base, radii.unsqueeze(-1)], dim=-1)
        center_radius_by_link[link_name] = center_radius_link
        center_radius_all_links.append(center_radius_link)

        for si in range(center_radius_link.shape[1]):
            center_radius_by_sphere[f"{link_name}[{si}]"] = center_radius_link[
                :, si, :
            ]

    center_radius_all = torch.cat(center_radius_all_links, dim=1)
    return center_radius_by_link, center_radius_by_sphere, center_radius_all


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

    tm_shelf = UrdfTransformManager()
    tm_shelf.add_transform("base_link", "world", np.eye(4))

    position = [0, 0.75, 0]
    quat = [0, 0, 1, 0]
    A2B = np.eye(4)
    A2B[:3, :3] = R.from_quat(quat).as_matrix()
    A2B[:3, 3] = np.asarray(position, dtype=float)
    URDF_shelf_0 = URDF_shelf.replace("base", "base_0")
    tm_shelf.load_urdf(URDF_shelf_0)
    tm_shelf.add_transform("base_0", "base_link", A2B)
    tm_shelf.plot_collision_objects("base_link")
    tm_shelf.plot_visuals("base_link")

    pos2 = [0, -0.75, 0]
    quat2 = [0, 0, 0, 1]
    A2B2 = np.eye(4)
    A2B2[:3, :3] = R.from_quat(quat2).as_matrix()
    A2B2[:3, 3] = np.asarray(pos2, dtype=float)
    URDF_shelf_1 = URDF_shelf.replace("base", "base_1")
    tm_shelf.load_urdf(URDF_shelf_1)
    tm_shelf.add_transform("base_1", "base_link", A2B2)
    tm_shelf.plot_collision_objects("base_link")
    tm_shelf.plot_visuals("base_link")

    pos3 = [0.75, 0, 0]
    quat3 = [0, 0, 0.5, 0.5]
    A2B3 = np.eye(4)
    A2B3[:3, :3] = R.from_quat(quat3).as_matrix()
    A2B3[:3, 3] = np.asarray(pos3, dtype=float)
    URDF_shelf_2 = URDF_shelf.replace("base", "base_2")
    tm_shelf.load_urdf(URDF_shelf_2)
    tm_shelf.add_transform("base_2", "base_link", A2B3)
    tm_shelf.plot_collision_objects("base_link")
    tm_shelf.plot_visuals("base_link")

    # 7 walls per shelf x 3 shelves = 21 collision objects total
    co = tm_shelf.collision_objects
    print(f"==>> len(co): {len(co)}")

    for b in co:
        print(b.frame)
        A2B = tm_shelf.get_transform(b.frame, "base_link")
        print(f"==>> A2B: \n{A2B}")
        print(b.size)
    plt.show()

    ax = make_3d_axis(ax_s=1.0)
    plot_transform(ax=ax, A2B=np.eye(4), s=0.3, lw=2, name="base_link")

    for b in co:
        A2B = tm_shelf.get_transform(b.frame, "base_link")
        size = b.size
        plot_box(ax=ax, A2B=A2B, size=size, wireframe=True, alpha=0.3)
    plt.show()

    raise
    import pytorch_kinematics as pk
    import yaml

    with open(
        os.path.join(rsrc, "./ur5e/ur5e_extract_calibrated_spherized.urdf"), "rb"
    ) as f:
        URDFrb = f.read()

    with open(
        os.path.join(rsrc, "./ur5e/ur5e_extract_calibrated_spherized.yml"), "r"
    ) as f:
        sphere_info = yaml.safe_load(f)

    chain = pk.build_serial_chain_from_urdf(
        URDFrb, root_link_name="base_link", end_link_name="tool0"
    )

    d = "cuda" if torch.cuda.is_available() else "cpu"
    chain = chain.to(device=d)
    print(f"==>> chain: \n{chain}")
    joint_names = chain.get_joint_parameter_names()
    print(f"==>> joint_names: \n{joint_names}")

    Q = torch.rand(1000, 6).to(d) * torch.pi - torch.pi / 2
    ret = chain.forward_kinematics(Q, end_only=False)
    sphere_link = sphere_info["metadata"]["links"]
    collision_sphere = sphere_info["collision_spheres"]
    sphere_link_count = []
    for link_name, sphere in collision_sphere.items():
        sphere_link_count.append(len(sphere))
    print(f"==>> sphere_link: \n{sphere_link}")
    print(f"==>> sphere_link_count: \n{sphere_link_count}")

    center_radius_by_link, center_radius_by_sphere, center_radius_all = (
        build_sphere_centers_from_fk(ret, collision_sphere, d)
    )
    print(f"==>> center_radius_all.shape: {center_radius_all.shape}")

    config_idx = 0
    center_radius_cfg = center_radius_all[config_idx].detach().cpu().numpy()
    print(f"==>> center_radius_cfg.shape: {center_radius_cfg.shape}")

    ax = make_3d_axis(ax_s=1.0)
    plot_transform(ax=ax, A2B=np.eye(4), s=0.3, lw=2, name="base_link")
    for x, y, z, r in center_radius_cfg:
        plot_sphere(
            ax=ax,
            p=[x, y, z],
            radius=r,
            alpha=0.3,
        )
    ax.set_box_aspect([1, 1, 1])
    plt.show()
