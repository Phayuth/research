import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import tqdm

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class Spatial3R:

    def __init__(self):
        self.links = [1, 1, 1]

    def fk_link(self, q):
        # base
        p0 = np.array([0, 0, 0])

        # first planar joint in local plane
        r1 = self.links[0] * np.cos(q[1])
        z1 = self.links[0] * np.sin(q[1])
        p1 = np.array([r1 * np.cos(q[0]), r1 * np.sin(q[0]), z1])

        # end-effector
        r2 = self.links[0] * np.cos(q[1]) + self.links[1] * np.cos(q[1] + q[2])
        z2 = self.links[0] * np.sin(q[1]) + self.links[1] * np.sin(q[1] + q[2])
        p2 = np.array([r2 * np.cos(q[0]), r2 * np.sin(q[0]), z2])

        return np.array([p0, p1, p2])

    def ik(self, T):
        pass


class RobotScene:

    def __init__(self, robot, obstacles):
        self.robot = robot
        self.wspace_limits = [(-2, 2), (-2, 2), (0, 2)]
        self.boxsize = {
            "box1": ((0.5, 0.5, 1), (1.5, 0, 0)),
            "box2": ((0.5, 0.5, 1), (-1.5, 0, 0)),
            "box3": ((0.5, 0.5, 1), (0, 1.5, 0)),
            "box4": ((0.5, 0.5, 1), (0, -1.5, 0)),
            "box5": ((0.5, 0.5, 1), (0, 0, 1.5)),
            "box6": ((0.5, 0.5, 1), (0, 0, -1.5)),
        }
        self.obstacles = []
        for name, (size, pos) in self.boxsize.items():
            colbox = trimesh.creation.box(extents=size)
            colbox.apply_translation(pos)
            self.obstacles.append(colbox)
        self.arm_thickness = 0.05

    def _link_cylinder(self, p0, p1):
        v = p1 - p0
        h = np.linalg.norm(v)
        if h < 1e-9:
            return None
        cyl = trimesh.creation.cylinder(
            radius=self.arm_thickness, height=h, sections=32
        )
        T = trimesh.geometry.align_vectors([0, 0, 1], v / h)
        cyl.apply_transform(T)
        cyl.apply_translation((p0 + p1) * 0.5)
        return cyl

    def arm_collision(self, q):
        fk_links = self.robot.fk_link(q)
        cyl1 = self._link_cylinder(fk_links[0], fk_links[1])
        cyl2 = self._link_cylinder(fk_links[1], fk_links[2])
        return [c for c in [cyl1, cyl2] if c is not None]

    def distance_to_obstacles(self, q):
        # Approximate penetration check by sampling points on arm links.
        arm_cylinders = self.arm_collision(q)
        for i, cyl in enumerate(arm_cylinders):
            for j, obs in enumerate(self.obstacles):
                pts, _ = trimesh.sample.sample_surface(cyl, 3000)
                sdf = obs.nearest.signed_distance(pts)
                print(
                    f"link {i}, obs {j}: max_penetration={sdf.max():.6f}, min_signed={sdf.min():.6f}"
                )

    def collision_check(self, q):
        # mesh to mesh fcl
        manager = trimesh.collision.CollisionManager()
        for k, obs in enumerate(self.obstacles):
            manager.add_object(f"obs_{k}", obs)
        for i, link_mesh in enumerate(self.arm_collision(q)):
            if manager.in_collision_single(link_mesh):
                return True
        return False

    def cspace_dataset_collision(self):
        num_samples = 360
        q1 = np.linspace(-np.pi, np.pi, num_samples)
        q2 = np.linspace(-np.pi, np.pi, num_samples)
        q3 = np.linspace(-np.pi, np.pi, num_samples)
        Q1, Q2, Q3 = np.meshgrid(q1, q2, q3, indexing="ij")
        joints = np.column_stack([Q1.ravel(), Q2.ravel(), Q3.ravel()])
        print("Generated joint samples:", joints.shape)
        col = np.empty(joints.shape[0])
        for i in tqdm.tqdm(range(joints.shape[0])):
            q = joints[i]
            col[i] = self.collision_check(q)
        joint_dataset = np.column_stack([joints, col])
        np.save(os.path.join(rsrc, "spatial3r_cspace.npy"), joint_dataset)

    def show_env(self, q):
        # supports
        plane = trimesh.creation.box(extents=(5, 5, 0.01))
        plane.visual.face_colors = [200, 200, 200, 80]
        axis = trimesh.creation.axis(origin_size=0.05, axis_length=2.5)
        box = trimesh.creation.box(extents=(5, 5, 5))
        box.visual.face_colors = [100, 150, 255, 40]

        # arm links
        arm_cylinders = self.arm_collision(q)

        # scene setup
        scene = trimesh.Scene()
        scene.add_geometry(plane)
        scene.add_geometry(box)
        scene.add_geometry(axis)
        for cyl in arm_cylinders:
            scene.add_geometry(cyl)
        for colbox in self.obstacles:
            scene.add_geometry(colbox)
        scene.set_camera(angles=(0, 0, 0), distance=3.0, center=[0, 0, 0])
        scene.show()


if __name__ == "__main__":
    robot = Spatial3R()
    scene = RobotScene(robot, None)
    q = np.array([0.0, 0.0, 0.0])
    # scene.collision_check(q)
    # scene.distance_to_obstacles(q)
    scene.cspace_dataset_collision()
    scene.show_env(q)
