import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from eaik.IK_DH import DhRobot
from pytransform3d.transform_manager import TransformManager
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis, plot_sphere, plot_box
import pytorch_kinematics as pk
import yaml
import torch

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
dir_rsrc = os.environ["RSRC_DIR"]
dir_urdf = os.path.join(dir_rsrc, "urdfs")
dir_rtsp = os.path.join(dir_rsrc, "rtsp_env")
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
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.robot_kin = RobotUR5eKin()
        self.chain = self.load_robot_chain()
        self.collision_sphere = self.load_robot_collision_spheres()
        self.box_in_base, self.boxsz_in_base = self.load_static_collision()

    def load_robot_chain(self):
        ur_sphr_ = os.path.join(
            dir_rsrc, "./ur5e/ur5e_extract_calibrated_spherized.urdf"
        )
        with open(ur_sphr_, "rb") as f:
            URDFrb = f.read()

        # build batch kinematic
        chain = pk.build_serial_chain_from_urdf(
            URDFrb,
            root_link_name="base_link",
            end_link_name="tool0",
        )
        chain = chain.to(device=self.device, dtype=self.dtype)
        # joint_names = chain.get_joint_parameter_names()
        # print(f"==>> joint_names: \n{joint_names}")
        return chain

    def load_robot_collision_spheres(self):
        sphr_info_ = os.path.join(
            dir_rsrc, "./ur5e/ur5e_extract_calibrated_spherized.yml"
        )
        with open(sphr_info_, "r") as f:
            sphere_info = yaml.safe_load(f)

        collision_sphere = sphere_info["collision_spheres"]
        return collision_sphere

    def load_static_collision(self):
        raise NotImplementedError(
            "load_static_collision must be defined in subclass"
        )

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
        Q = torch.as_tensor(Q, dtype=self.dtype, device=self.device)

        # batch forward kinematics
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
            tf_device = link_tf.device

            # [num_spheres, 3]
            centers_local = torch.tensor(
                [s["center"] for s in spheres],
                dtype=link_tf.dtype,
                device=tf_device,
            )
            # [num_spheres, 4]
            centers_hom = torch.cat(
                [
                    centers_local,
                    torch.ones(
                        centers_local.shape[0],
                        1,
                        dtype=link_tf.dtype,
                        device=tf_device,
                    ),
                ],
                dim=1,
            )

            centers_base = torch.einsum(
                "bij,nj->bni",
                link_tf,
                centers_hom,
            )[..., :3]

            radii = torch.empty(
                (num_ik, len(spheres)),
                dtype=link_tf.dtype,
                device=tf_device,
            )
            for si, s in enumerate(spheres):
                # Supports fixed radius or optional per-IK radii in s["radius_by_ik"].
                if "radius_by_ik" in s:
                    r = torch.as_tensor(
                        s["radius_by_ik"], dtype=link_tf.dtype, device=tf_device
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
            center_radius_all_links.append(center_radius_link)
            # center_radius_by_link[link_name] = center_radius_link

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
        view_elev = 20
        view_azim = 164
        view_roll = -4

        def as_numpy(value):
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        box_in_base = as_numpy(self.box_in_base)
        boxsz_in_base = as_numpy(self.boxsz_in_base)

        qSphere = self.fkin_sphere(q)
        sphere = qSphere[0].detach().cpu().numpy()

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
        for i in range(box_in_base.shape[0]):
            plot_box(
                ax=ax,
                A2B=box_in_base[i],
                size=boxsz_in_base[i],
                wireframe=False,
                alpha=0.3,
            )

        # task poses
        if Hlist is not None:
            for h in Hlist:
                plot_transform(ax=ax, A2B=h, s=0.1, name="task")

        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)
        plt.show()


class SceneUR5eSpherizedAirbusShopFloor(SceneUR5eSpherized):

    def __init__(self):
        super().__init__()
        self.box_in_base, self.boxsz_in_base = self.load_static_collision()

    def load_static_collision(self):
        ff = os.path.join(dir_rsrc, "urdfs", "airbus_shopfloor_collision.yaml")
        with open(ff, "r") as f:
            data = yaml.safe_load(f)

        n = len(data["collision_in_target_link"])
        box_in_base = np.zeros((n, 4, 4))
        boxsz_in_base = np.zeros((n, 3))

        collision_in_target_link = data["collision_in_target_link"]
        for i, (box_name, box_data) in enumerate(collision_in_target_link.items()):
            link_name = box_data["link"]
            center = np.array(box_data["center"])
            size = np.array(box_data["size"])
            H = np.eye(4)
            H[:3, 3] = center
            box_in_base[i] = H
            boxsz_in_base[i] = size

        print(box_in_base)
        print(boxsz_in_base)
        return box_in_base, boxsz_in_base


class SceneUR5eSpherizedSingleStool(SceneUR5eSpherized):

    def __init__(self):
        super().__init__()
        self.box_in_base, self.boxsz_in_base = self.load_static_collision()

    def load_static_collision(self):
        ff = os.path.join(dir_rsrc, "urdfs", "single_stool_collision.yaml")
        with open(ff, "r") as f:
            data = yaml.safe_load(f)

        n = len(data["collision_in_target_link"])
        box_in_base = np.zeros((n, 4, 4))
        boxsz_in_base = np.zeros((n, 3))

        collision_in_target_link = data["collision_in_target_link"]
        for i, (box_name, box_data) in enumerate(collision_in_target_link.items()):
            link_name = box_data["link"]
            center = np.array(box_data["center"])
            size = np.array(box_data["size"])
            H = np.eye(4)
            H[:3, 3] = center
            box_in_base[i] = H
            boxsz_in_base[i] = size

        print(box_in_base)
        print(boxsz_in_base)
        return box_in_base, boxsz_in_base


class SceneUR5eSpherizedThreePlanarBoard(SceneUR5eSpherized):

    def __init__(self):
        super().__init__()
        self.box_in_base, self.boxsz_in_base = self.load_static_collision()

    def load_static_collision(self):
        ff = os.path.join(dir_rsrc, "urdfs", "three_planar_board_collision.yaml")
        with open(ff, "r") as f:
            data = yaml.safe_load(f)

        n = len(data["collision_in_target_link"])
        box_in_base = np.zeros((n, 4, 4))
        boxsz_in_base = np.zeros((n, 3))

        collision_in_target_link = data["collision_in_target_link"]
        for i, (box_name, box_data) in enumerate(collision_in_target_link.items()):
            link_name = box_data["link"]
            center = np.array(box_data["center"])
            size = np.array(box_data["size"])
            H = np.eye(4)
            H[:3, 3] = center
            box_in_base[i] = H
            boxsz_in_base[i] = size

        print(box_in_base)
        print(boxsz_in_base)
        return box_in_base, boxsz_in_base


class SceneUR5eSpherizedSingleBarStrict(SceneUR5eSpherized):

    def __init__(self):
        super().__init__()
        self.box_in_base, self.boxsz_in_base = self.load_static_collision()

    def load_static_collision(self):
        ff = os.path.join(dir_rsrc, "urdfs", "single_bar_strict_collision.yaml")
        with open(ff, "r") as f:
            data = yaml.safe_load(f)

        n = len(data["collision_in_target_link"])
        box_in_base = np.zeros((n, 4, 4))
        boxsz_in_base = np.zeros((n, 3))

        collision_in_target_link = data["collision_in_target_link"]
        for i, (box_name, box_data) in enumerate(collision_in_target_link.items()):
            link_name = box_data["link"]
            center = np.array(box_data["center"])
            size = np.array(box_data["size"])
            H = np.eye(4)
            H[:3, 3] = center
            box_in_base[i] = H
            boxsz_in_base[i] = size

        print(box_in_base)
        print(boxsz_in_base)
        return box_in_base, boxsz_in_base


class SceneUR5eSpherizedThreeShelf(SceneUR5eSpherized):

    def __init__(self):
        super().__init__()
        self.box_in_base, self.boxsz_in_base = self.load_three_shelf_collision()

    def load_static_collision(self):
        ff = os.path.join(dir_rsrc, "urdfs", "three_shelf_collision.yaml")
        with open(ff, "r") as f:
            data = yaml.safe_load(f)

        n = len(data["collision_in_target_link"])
        box_in_base = np.zeros((n, 4, 4))
        boxsz_in_base = np.zeros((n, 3))

        collision_in_target_link = data["collision_in_target_link"]
        for i, (box_name, box_data) in enumerate(collision_in_target_link.items()):
            link_name = box_data["link"]
            center = np.array(box_data["center"])
            size = np.array(box_data["size"])
            H = np.eye(4)
            H[:3, 3] = center
            box_in_base[i] = H
            boxsz_in_base[i] = size

        print(box_in_base)
        print(boxsz_in_base)
        return box_in_base, boxsz_in_base


class SceneOMPLPlanner:

    def __init__(self, collision_checker):
        from ompl import base as ob
        from ompl import geometric as og
        from ompl import util as ou

        self.ob = ob
        self.og = og
        self.ou = ou

        ou.RNG.setSeed(42)
        self.collision_checker = collision_checker

        self.space = ob.RealVectorStateSpace(6)
        self.bounds = ob.RealVectorBounds(6)
        self.limit6 = [
            2 * np.pi,
            2 * np.pi,
            np.pi,
            2 * np.pi,
            2 * np.pi,
            2 * np.pi,
        ]
        for i in range(6):
            self.bounds.setLow(i, -self.limit6[i])
            self.bounds.setHigh(i, self.limit6[i])
        self.bounds.setLow(1, -np.pi)
        self.bounds.setHigh(1, 0)
        self.space.setBounds(self.bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )
        # self.planner = og.BITstar(self.ss.getSpaceInformation())
        self.planner = og.ABITstar(self.ss.getSpaceInformation())
        # self.planner = og.AITstar(self.ss.getSpaceInformation())
        # self.planner.setRange(0.1)
        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        q = [state[0], state[1], state[2], state[3], state[4], state[5]]
        # collision check return True if collision, False if free
        col = self.collision_checker(q).detach().cpu().numpy().item()
        return not col

    def query_planning(self, start_list, goal_list):
        # Important!
        # Clear previous planning data to ensure fresh planning because caching
        self.ss.clear()

        start = self.ob.State(self.space)
        start[0] = start_list[0]
        start[1] = start_list[1]
        start[2] = start_list[2]
        start[3] = start_list[3]
        start[4] = start_list[4]
        start[5] = start_list[5]
        goal = self.ob.State(self.space)
        goal[0] = goal_list[0]
        goal[1] = goal_list[1]
        goal[2] = goal_list[2]
        goal[3] = goal_list[3]
        goal[4] = goal_list[4]
        goal[5] = goal_list[5]

        dist = np.linalg.norm(np.array(goal_list) - np.array(start_list))

        self.ss.setStartAndGoalStates(start, goal)
        status = self.ss.solve(10.0)
        print("Plan from ", start_list, " to ", goal_list, "estimate cost:", dist)
        (
            print("EXACT")
            if status.getStatus() == status.EXACT_SOLUTION
            else print("Invalid result")
        )
        if status.getStatus() == status.EXACT_SOLUTION:
            self.ss.simplifySolution()
            path = self.ss.getSolutionPath()
            path_cost = path.length()

            print("Found solution:")
            print(f"Path cost: {path_cost}")
            print(self.ss.getSolutionPath())

            pathlist = []
            for i in range(path.getStateCount()):
                pi = path.getState(i)
                pathlist.append([pi[0], pi[1], pi[2], pi[3], pi[4], pi[5]])
            return pathlist, path_cost
        else:
            print("No solution found")
            return None


def load_taskspace_poses():
    fyaml = os.path.join(dir_rtsp, "three_shelf_taskspace_poses.yaml")
    with open(fyaml, "r") as f:
        data = yaml.safe_load(f)

if __name__ == "__main__":
    robot_kin = RobotUR5eKin()
    q = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
    ax = robot_kin.plot_link_transforms(q)
    plt.show()
