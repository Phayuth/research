import os
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class Spatial3R:

    def __init__(self):
        self.links = [1, 1, 1]

    def fk(self, q):
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
        self.obstacles = obstacles

    def show_env(self, q):
        pass


if __name__ == "__main__":
    robot = Spatial3R()
    scene = RobotScene(robot, None)
    q = np.array([0.5, 0.5, 0.5])
    fk = robot.fk(q)

    import trimesh

    radius = 0.05
    l1_vec = fk[1] - fk[0]
    l2_vec = fk[2] - fk[1]
    cyl1 = trimesh.creation.cylinder(radius=radius, height=1)
    cyl2 = trimesh.creation.cylinder(radius=radius, height=1)
    plane = trimesh.creation.box(extents=(2, 2, 0.01))
    T = trimesh.geometry.align_vectors([0, 0, 1], l1_vec)
    cyl1.apply_transform(T)
    cyl1.apply_translation((fk[0] + fk[1]) / 2)
    T = trimesh.geometry.align_vectors([0, 0, 1], l2_vec)
    cyl2.apply_transform(T)
    cyl2.apply_translation((fk[1] + fk[2]) / 2)
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=0.8)

    # ---- combine into a scene ----
    scene = trimesh.Scene()
    scene.add_geometry(axis)
    scene.add_geometry(cyl1)
    scene.add_geometry(cyl2)
    scene.add_geometry(plane)
    scene.set_camera(angles=(0, 0, 0), distance=3.0, center=[0, 0, 0])
    # ---- show viewer ----
    scene.show()
