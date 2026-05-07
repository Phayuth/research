import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from eaik.IK_DH import DhRobot
from pytransform3d.transform_manager import TransformManager
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.plot_utils import plot_sphere, plot_box
import torch
import pytorch_kinematics as pk
import yaml

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]
device = "cuda" if torch.cuda.is_available() else "cpu"


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

        # A_chain, T_chain = robot_kin.plot_link_transforms(q)
        # for i, A_i in enumerate(A_chain, start=1):
        #     print(f"A_{i} (link_{i-1} -> link_{i}):\n", A_i)
        # for i, T_0i in enumerate(T_chain):
        #     print(f"T_0{i} (base -> link_{i}):\n", T_0i)

        # return relative_tf, world_tf

        return ax


class SceneUR5eSpherized:

    def __init__(self):
        self.robot_kin = RobotUR5eKin()
        self.chain = self.load_urdf_chain()
        self.collision_sphere = self.load_spheres()
        self.box_in_base, self.boxsz_in_base = self.load_boxes_to_base_link()

    def load_urdf_chain(self):
        ur_sphr_ = os.path.join(
            rsrc, "./ur5e/ur5e_extract_calibrated_spherized.urdf"
        )
        # with open(ur_sphr_, "r") as f:
        #     URDF = f.read()
        # tm = UrdfTransformManager()
        # tm.load_urdf(URDF)
        # tm.set_joint("shoulder_pan_joint", q[0])
        # tm.set_joint("shoulder_lift_joint", q[1])
        # tm.set_joint("elbow_joint", q[2])
        # tm.set_joint("wrist_1_joint", q[3])
        # tm.set_joint("wrist_2_joint", q[4])
        # tm.set_joint("wrist_3_joint", q[5])
        # FKurdf = tm.get_transform("tool0", "base_link")
        # print(f"==>> FKurdf: \n{FKurdf}")

        with open(ur_sphr_, "rb") as f:
            URDFrb = f.read()

        # build batch kinematic
        chain = pk.build_serial_chain_from_urdf(
            URDFrb,
            root_link_name="base_link",
            end_link_name="tool0",
        )
        # joint_names = chain.get_joint_parameter_names()
        # print(f"==>> joint_names: \n{joint_names}")
        return chain

    def load_spheres(self):
        sphr_info_ = os.path.join(
            rsrc, "./ur5e/ur5e_extract_calibrated_spherized.yml"
        )
        with open(sphr_info_, "r") as f:
            sphere_info = yaml.safe_load(f)

        sphere_link = sphere_info["metadata"]["links"]
        collision_sphere = sphere_info["collision_spheres"]
        sphere_link_count = []
        for link_name, sphere in collision_sphere.items():
            sphere_link_count.append(len(sphere))
        # print(f"==>> sphere_link: \n{sphere_link}")
        # print(f"==>> sphere_link_count: \n{sphere_link_count}")
        return collision_sphere

    def load_boxes_to_base_link(self):
        # tf to transform and prefix for each shelf in the world frame
        shelf_tf = {
            0: ((0, 0.75, 0), (0, 0, 1, 0), "id_0_"),
            1: ((0, -0.75, 0), (0, 0, 0, 1), "id_1_"),
            2: ((0.75, 0, 0), (0, 0, 0.5, 0.5), "id_2_"),
        }

        tm_shelf = UrdfTransformManager()
        tm_shelf.add_transform("base_link", "world", np.eye(4))

        for id, (position, quat, prefix) in shelf_tf.items():
            position = np.asarray(position, dtype=float)
            quat = np.asarray(quat, dtype=float)
            A2B = np.eye(4)
            A2B[:3, :3] = R.from_quat(quat).as_matrix()
            A2B[:3, 3] = position
            urdf_path = os.path.join(rsrc, f"./ur5e/shelf.urdf")
            with open(urdf_path, "r") as f:
                urdf_str = f.read()
            urdf_str = self.add_prefix_to_urdf(urdf_str, prefix)
            tm_shelf.load_urdf(urdf_str)
            tm_shelf.add_transform(f"{prefix}base", "base_link", A2B)

        plane_ = os.path.join(rsrc, "./ur5e/plane.urdf")
        with open(plane_, "r") as f:
            plane_urdf = f.read()
        tm_shelf.load_urdf(plane_urdf)
        tm_shelf.add_transform("plane", "base_link", np.eye(4))

        # collecting collsion objects from all shelves
        # 7 walls per shelf x 3 shelves = 21 collision objects total
        col_obj = tm_shelf.collision_objects
        col_obj_count = len(col_obj)
        box_in_local = np.empty((col_obj_count, 4, 4))
        boxsz_in_local = np.empty((col_obj_count, 3))
        for i, b in enumerate(col_obj):
            A2B = tm_shelf.get_transform(b.frame, "base_link")
            size = b.size
            box_in_local[i] = A2B
            boxsz_in_local[i] = size
        box_in_base = np.empty_like(box_in_local)
        boxsz_in_base = np.empty_like(boxsz_in_local)
        for i in range(box_in_local.shape[0]):
            box_in_base[i], boxsz_in_base[i] = self.box_to_base_link(
                box_in_local[i], boxsz_in_local[i]
            )

        return box_in_base, boxsz_in_base

    @staticmethod
    def add_prefix_to_urdf(urdf_str, prefix="id_0_"):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(urdf_str)

        # helper: prefix if not already prefixed
        def pref(name):
            if name is None:
                return None
            return name if name.startswith(prefix) else prefix + name

        # robot name
        if "name" in root.attrib:
            root.attrib["name"] = pref(root.attrib["name"])

        # link names
        for link in root.findall("link"):
            link.attrib["name"] = pref(link.attrib["name"])

        # joint names + parent/child refs
        for joint in root.findall("joint"):
            joint.attrib["name"] = pref(joint.attrib["name"])

            parent = joint.find("parent")
            if parent is not None:
                parent.attrib["link"] = pref(parent.attrib["link"])

            child = joint.find("child")
            if child is not None:
                child.attrib["link"] = pref(child.attrib["link"])

        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def box_to_base_link(A2B, size):
        """
        Convert an oriented box into an axis-aligned box in base_link.

        Parameters
        ----------
        A2B : (4, 4) array
            Box frame pose in base_link.
        size : (3,) array
            Box dimensions in the box's local frame: [x, y, z].

        Returns
        -------
        A2B_new : (4, 4) array
            Axis-aligned box pose in base_link, with identity rotation.
        size_new : (3,) array
            Axis-aligned extents in base_link.
        """
        size = np.asarray(size, dtype=float)
        half = 0.5 * size

        corners_local = np.array(
            [
                [sx * half[0], sy * half[1], sz * half[2], 1.0]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )

        corners_base = (A2B @ corners_local.T).T[:, :3]

        mn = corners_base.min(axis=0)
        mx = corners_base.max(axis=0)

        center = 0.5 * (mn + mx)
        size_new = mx - mn

        A2B_new = np.eye(4)
        A2B_new[:3, 3] = center

        return A2B_new, size_new

    def fkin_sphere(self, Q):
        """
        Compute collision sphere centers for a batch of joint configurations.

        Parameters
        ----------
        chain : pytorch_kinematics.Chain
            The robot kinematic chain.
        Q : (ntraj, dof) tensor
            Batch of joint configurations.
        collision_sphere : dict[link_name] -> list of spheres
            Sphere definitions per link, where each sphere has a "center" and "radius".

        Returns
        -------
        spheres_in_base : (ntraj, total_num_spheres, 4) tensor
            Sphere centers in base frame and their radii for all trajectories.
        """
        fk_tf_dict = self.chain.forward_kinematics(Q, end_only=False)

        # Transform link-local collision spheres to base frame for batched FK.
        # spheres_in_base = fkin_sphere_links(fk_tf_dict, self.collision_sphere, Q.device)
        first_link_name = next(iter(self.collision_sphere))
        first_tf = fk_tf_dict[first_link_name].get_matrix()
        num_ik = first_tf.shape[0]

        # center_radius_by_link = {}
        # center_radius_by_sphere = {}
        center_radius_all_links = []

        for link_name, spheres in self.collision_sphere.items():
            link_tf = fk_tf_dict[link_name].get_matrix()  # [num_ik, 4, 4]

            # [num_spheres, 3]
            centers_local = torch.tensor(
                [s["center"] for s in spheres], dtype=link_tf.dtype, device=device
            )
            # [num_spheres, 4]
            centers_hom = torch.cat(
                [
                    centers_local,
                    torch.ones(
                        centers_local.shape[0],
                        1,
                        dtype=link_tf.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )

            centers_base = torch.einsum("bij,nj->bni", link_tf, centers_hom)[
                ..., :3
            ]

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

            center_radius_link = torch.cat(
                [centers_base, radii.unsqueeze(-1)], dim=-1
            )
            # center_radius_by_link[link_name] = center_radius_link
            center_radius_all_links.append(center_radius_link)

            # for si in range(center_radius_link.shape[1]):
            #     center_radius_by_sphere[f"{link_name}[{si}]"] = center_radius_link[
            #         :, si, :
            #     ]

        spheres_in_base = torch.cat(center_radius_all_links, dim=1)
        return spheres_in_base

    def check_sphr_box_vec_torch(self, spheres):
        """
        Torch version of the batched sphere-box collision check.

        Parameters
        ----------
        spheres : (ntraj, nsphere, 4) tensor-like
            Sphere centers xyz and radius.
        boxes : (nbox, 4, 4) or (nbox, 3) tensor-like
            Box transforms or box centers.
        boxes_size : (nbox, 3) tensor-like
            Box sizes [sx, sy, sz] from the center.

        Returns
        -------
        collision : (ntraj,) bool tensor
            Whether each trajectory collides with any box.
        """
        spheres = torch.as_tensor(spheres)
        self.box_in_base = torch.as_tensor(
            self.box_in_base, device=spheres.device, dtype=spheres.dtype
        )
        self.boxsz_in_base = torch.as_tensor(
            self.boxsz_in_base, device=spheres.device, dtype=spheres.dtype
        )

        sphere_center = spheres[..., :3]
        sphere_radius = spheres[..., 3]

        if self.box_in_base.ndim == 3 and self.box_in_base.shape[-2:] == (4, 4):
            box_center = self.box_in_base[:, :3, 3]
        elif self.box_in_base.ndim == 2 and self.box_in_base.shape[1] == 3:
            box_center = self.box_in_base
        else:
            raise ValueError(
                "boxes must be either (nbox, 3) centers or (nbox, 4, 4) transforms"
            )

        half_size = 0.5 * self.boxsz_in_base
        if half_size.ndim == 1:
            half_size = half_size[None, :]

        delta = sphere_center[:, :, None, :] - box_center[None, None, :, :]
        closest = torch.clamp(
            delta,
            min=-half_size[None, None, :, :],
            max=half_size[None, None, :, :],
        )
        sep = delta - closest

        dist_sq = torch.sum(sep * sep, dim=-1)
        r_sq = sphere_radius[:, :, None] ** 2
        collision_per_pair = dist_sq <= r_sq
        return collision_per_pair.any(dim=(1, 2))

    def collision_check(self, Q):
        spheres_in_base = self.fkin_sphere(Q)
        return self.check_sphr_box_vec_torch(spheres_in_base)

    def view_scene(self, q, Hlist=None):
        qSphere = self.fkin_sphere(q)
        sphere = qSphere[0]

        ax = make_3d_axis(ax_s=1.0)
        plot_transform(ax=ax, A2B=np.eye(4), s=0.3, lw=2, name="base_link")

        # robot spheres
        for x, y, z, r in sphere:
            plot_sphere(
                ax=ax,
                p=[x, y, z],
                radius=r,
                alpha=0.3,
                color="r",
            )

        # boxes
        for i in range(self.box_in_base.shape[0]):
            plot_box(
                ax=ax,
                A2B=self.box_in_base[i],
                size=self.boxsz_in_base[i],
                wireframe=False,
                alpha=0.3,
            )

        # task poses
        if Hlist is not None:
            for h in Hlist:
                plot_transform(ax=ax, A2B=h, s=0.1, name="task")

        ax.set_box_aspect([1, 1, 1])
        plt.show()


def pick_task_poses():
    def _gen_linear_H(s, e, quat, num_tasks=10):
        t = np.linspace(s, e, num_tasks)
        Hlist = [np.eye(4) for _ in range(num_tasks)]
        for i in range(num_tasks):
            Hlist[i][:3, 3] = t[i]
            Hlist[i][:3, :3] = R.from_quat(quat).as_matrix()
        return Hlist

    def _Hrot_Z(a):
        H = np.eye(4)
        c, s = np.cos(a), np.sin(a)
        H[0:3, 0:3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        return H

    def _RotPI(H):
        Hdh_to_urdf = _Hrot_Z(np.pi)
        return np.linalg.inv(Hdh_to_urdf) @ H

    size = 4
    params = {
        0: ([-0.4, 0.6, 0.5], [0.4, 0.6, 0.5], [-0.707106, 0.0, 0.0, 0.707106]),
        1: ([-0.4, 0.6, 0.2], [0.4, 0.6, 0.2], [-0.707106, 0.0, 0.0, 0.707106]),
        2: ([-0.6, -0.4, 0.5], [-0.6, 0.4, 0.5], [-0.5, -0.5, 0.5, 0.5]),
        3: ([-0.6, -0.4, 0.2], [-0.6, 0.4, 0.2], [-0.5, -0.5, 0.5, 0.5]),
        4: ([0.4, -0.6, 0.5], [-0.4, -0.6, 0.5], [0.0, -0.707106, 0.707106, 0.0]),
        5: ([0.4, -0.6, 0.2], [-0.4, -0.6, 0.2], [0.0, -0.707106, 0.707106, 0.0]),
    }
    HH = []
    for k in params:
        s, e, quat = params[k]
        quat_noise = quat + np.random.normal(0, 0.05, size=4)
        HH += _gen_linear_H(s, e, quat_noise, num_tasks=size)
    Hlist = np.array(HH)
    Hlist = np.array([_RotPI(H) for H in Hlist])
    return Hlist


def batch_check():
    robot_kin = RobotUR5eKin()
    scene = SceneUR5eSpherized()

    q = torch.tensor(
        [0, -np.pi / 4, 0, -np.pi / 2, 0, np.pi / 3], dtype=torch.float32
    ).to(device)
    Hlist = pick_task_poses()
    scene.view_scene(q, Hlist)

    ntraj = 100000
    dof = 6
    Q = torch.rand(ntraj, dof).to(device) * torch.pi - torch.pi / 2
    col_states = scene.collision_check(Q)
    collision_count = col_states.sum().item()
    print(f"==>> Total trajectories: {ntraj}")
    print(f"==>> Colliding trajectories: {collision_count}")
    print(f"==>> Collision rate: {collision_count / ntraj * 100:.2f}%")

    nstore = 1000000
    dataset = torch.empty(nstore, dof).to(device)
    dataset_y = torch.empty(nstore, dtype=torch.bool).to(device)
    batch = 100000
    it = nstore // batch
    for i in tqdm.tqdm(range(it)):
        start = i * batch
        end = start + batch
        dataset[start:end, :dof] = (
            torch.rand(batch, dof).to(device) * torch.pi - torch.pi / 2
        )
        col_states = scene.collision_check(dataset[start:end, :dof])
        dataset_y[start:end] = col_states

    collision_rate = dataset_y.sum().item() / nstore
    print(f"==>> Dataset collision rate: {collision_rate * 100:.2f}%")


if __name__ == "__main__":
    batch_check()
