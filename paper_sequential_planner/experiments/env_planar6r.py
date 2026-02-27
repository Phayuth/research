import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box, Polygon, MultiPolygon
from shapely.ops import nearest_points


np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class Planar6R:

    def __init__(self):
        self.link_lengths = np.ones(6)  # Assuming all link lengths are 1 unit

    def fk_vectorized(self, joint_angles):
        n, num_joints = joint_angles.shape
        cumulative_angles = np.cumsum(joint_angles, axis=1)
        x_displacements = self.link_lengths * np.cos(cumulative_angles)
        y_displacements = self.link_lengths * np.sin(cumulative_angles)
        displacements = np.stack((x_displacements, y_displacements), axis=-1)
        positions = np.cumsum(displacements, axis=1)
        # base shape (n, 1, 2), to represent the base at (0, 0)
        base = np.zeros((n, 1, 2))
        result = np.concatenate([base, positions], axis=1)  # Shape (n, 7, 2)
        return result


class RobotScene:

    def __init__(self, robot, obstacles):
        self.robot = robot
        # o1 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])
        # o2 = Polygon([(-4, 3), (-3, 3), (-3, 4), (-4, 4)])
        # oo = MultiPolygon([o1, o2])
        o1 = Polygon([(-6, -6), (-2, -6), (-2, 6), (-6, 6)])
        self.arm_thinkness = 0.2
        self.obstacles = MultiPolygon([o1])

    def distance_to_obstacles(self, q):
        links_xy = self.robot.fk_vectorized(q[np.newaxis, :])[0]
        armcols = [LineString(pp).buffer(self.arm_thinkness) for pp in links_xy]
        dmin = np.array([armcol.distance(self.obstacles) for armcol in armcols])
        best_idx = np.argmin(dmin)
        return {"distance": dmin[best_idx], "link_index": best_idx}, dmin

    def get_dmin(self, links_xy):
        dmin = (
            LineString(links_xy)
            .buffer(self.arm_thinkness)
            .distance(self.obstacles)
        )
        return dmin

    def collision_check(self, q):
        links_xy = self.robot.fk_vectorized(q[np.newaxis, :])[0]
        dmin = self.get_dmin(links_xy)
        if dmin <= 0:
            return True
        else:
            return False

    def show_env(self, q):
        links_xy = self.robot.fk_vectorized(q[np.newaxis, :])[0]
        armcols = LineString(links_xy).buffer(self.arm_thinkness)
        dmin = self.get_dmin(links_xy)
        armcols_nearest, oo_nearest = nearest_points(armcols, self.obstacles)

        fig, ax = plt.subplots()

        # arm links
        ax.plot(links_xy[:, 0], links_xy[:, 1], "-o", color="blue")
        x, y = armcols.exterior.xy
        ax.fill(x, y, alpha=0.5, fc="green", ec="black")

        # obstacles
        for poly in self.obstacles.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="red", ec="black")

        # nearest points and distance line
        ax.plot(
            [oo_nearest.x, armcols_nearest.x],
            [oo_nearest.y, armcols_nearest.y],
            "o--",
            color="purple",
            markersize=12,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.grid()
        plt.show()

    def generate_grid_sample(self):
        num_samples = 20
        q1 = np.linspace(-np.pi, np.pi, num_samples)
        q2 = np.linspace(-np.pi, np.pi, num_samples)
        q3 = np.linspace(-np.pi, np.pi, num_samples)
        q4 = np.linspace(-np.pi, np.pi, num_samples)
        q5 = np.linspace(-np.pi, np.pi, num_samples)
        q6 = np.linspace(-np.pi, np.pi, num_samples)
        Q1, Q2, Q3, Q4, Q5, Q6 = np.meshgrid(q1, q2, q3, q4, q5, q6, indexing="ij")
        joint_dataset = np.column_stack(
            [
                Q1.ravel(),
                Q2.ravel(),
                Q3.ravel(),
                Q4.ravel(),
                Q5.ravel(),
                Q6.ravel(),
            ]
        )
        print(joint_dataset)
        print(joint_dataset.shape)  # (64_000_000, 6)
        return joint_dataset


if __name__ == "__main__":
    robot = Planar6R()
    scene = RobotScene(robot, None)
    q = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
    scene.show_env(q)
