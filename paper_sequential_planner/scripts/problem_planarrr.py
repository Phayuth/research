import os
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class PlanarRR:

    def __init__(self):
        self.a1 = 2
        self.a2 = 2

    def forward_kinematic(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        link_end_pose = []
        link_end_pose.append([0, 0])

        # link 1 pose
        x1 = self.a1 * np.cos(theta1)
        y1 = self.a1 * np.sin(theta1)
        link_end_pose.append([x1, y1])

        # link 2 pose
        x2 = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
        y2 = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)
        link_end_pose.append([x2, y2])

        return link_end_pose

    def inverse_kinematic(self, X):
        x = X[0, 0]
        y = X[1, 0]

        # check if the desired pose is inside of task space or not
        rd = np.sqrt(x**2 + y**2)
        link_length = self.a1 + self.a2
        if rd > link_length:
            print("The desired pose is outside of taskspace")
            return None

        elbow_down = -1
        elbow_up = 1

        D = (x**2 + y**2 - self.a1**2 - self.a2**2) / (2 * self.a1 * self.a2)
        theta2_down = np.arctan2(elbow_down * np.sqrt(1 - D**2), D)
        theta2_up = np.arctan2(elbow_up * np.sqrt(1 - D**2), D)
        theta1_down = np.arctan2(y, x) - np.arctan2(
            (self.a2 * np.sin(theta2_down)),
            (self.a1 + self.a2 * np.cos(theta2_down)),
        )
        theta1_up = np.arctan2(y, x) - np.arctan2(
            (self.a2 * np.sin(theta2_up)), (self.a1 + self.a2 * np.cos(theta2_up))
        )
        return (
            np.array([theta1_down, theta2_down]),
            np.array([theta1_up, theta2_up]),
        )


class RobotScene:

    def __init__(self, robot, obstacles):
        self.robot = robot
        self.obstacles = obstacles

    def robot_collision_links(self, theta):
        link_points = self.robot.forward_kinematic(theta)
        link_shapes = []
        for i in range(len(link_points) - 1):
            p1 = tuple(link_points[i])
            p2 = tuple(link_points[i + 1])
            link_shapes.append(LineString([p1, p2]))
        return link_shapes

    def distance_to_obstacles(self, theta):
        links = self.robot_collision_links(theta)
        results = []
        for li, link in enumerate(links):
            for sj, shp in enumerate(self.obstacles):
                dist = link.distance(shp)
                p_link, p_shape = nearest_points(link, shp)
                results.append(
                    {
                        "link_idx": li,
                        "shape_idx": sj,
                        "distance": float(dist),
                        "link_point": (float(p_link.x), float(p_link.y)),
                        "shape_point": (float(p_shape.x), float(p_shape.y)),
                    }
                )

        if not results:
            return None, []

        best = min(results, key=lambda r: r["distance"])
        return best, results

    def cspace_obstacles(self, generate=False, save=False, plot=False):
        if generate:
            print("Generating C-space obstacle plot...")
            num_samples = 360
            theta1_samples = np.linspace(-np.pi, np.pi, num_samples)
            theta2_samples = np.linspace(-np.pi, np.pi, num_samples)
            cspace_obs = []

            for i in range(num_samples):
                for j in range(num_samples):
                    theta = np.array([[theta1_samples[i]], [theta2_samples[j]]])
                    best, _ = self.distance_to_obstacles(theta)
                    if best is not None and best["distance"] <= 0.0:
                        cspace_obs.append((theta1_samples[i], theta2_samples[j]))
            cspace_obs = np.array(cspace_obs)

        if save:
            np.save(os.path.join(rsrc, "cspace_obstacles.npy"), cspace_obs)

        if plot:
            cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
            fig, ax = plt.subplots()
            ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
            ax.set_aspect("equal", "box")
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            return ax

    def cspace_dataset_collision(self):
        print("Generating C-space dataset with collision labels...")
        num_samples = 360
        theta1_samples = np.linspace(-np.pi, np.pi, num_samples)
        theta2_samples = np.linspace(-np.pi, np.pi, num_samples)
        dataset = []
        for i in range(num_samples):
            for j in range(num_samples):
                theta = np.array([[theta1_samples[i]], [theta2_samples[j]]])
                best, _ = self.distance_to_obstacles(theta)
                if best is not None and best["distance"] <= 0.0:
                    dataset.append((theta1_samples[i], theta2_samples[j], 1))
                else:
                    dataset.append((theta1_samples[i], theta2_samples[j], -1))
        dataset = np.array(dataset)
        np.save(os.path.join(rsrc, "cspace_dataset.npy"), dataset)

    def cspace_dataset_nearest_distance(self):
        print("Generating C-space dataset with nearest distance...")
        num_samples = 360
        theta1_samples = np.linspace(-np.pi, np.pi, num_samples)
        theta2_samples = np.linspace(-np.pi, np.pi, num_samples)
        dataset = []
        for i in range(num_samples):
            for j in range(num_samples):
                theta = np.array([[theta1_samples[i]], [theta2_samples[j]]])
                best, _ = self.distance_to_obstacles(theta)
                if best is not None:
                    dataset.append(
                        (theta1_samples[i], theta2_samples[j], best["distance"])
                    )
                else:
                    dataset.append((theta1_samples[i], theta2_samples[j], np.inf))
        dataset = np.array(dataset)
        np.save(os.path.join(rsrc, "cspace_dataset_nearest_distance.npy"), dataset)

    def plot(self, theta):
        links = self.robot_collision_links(theta)
        best, results = self.distance_to_obstacles(theta)

        fig, ax = plt.subplots()
        for link in links:
            x, y = link.xy
            ax.plot(x, y, color="blue", linewidth=4, solid_capstyle="round")

        for shp in self.obstacles:
            x, y = shp.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="red", ec="black")

        if best is not None:
            lp = best["link_point"]
            sp = best["shape_point"]
            ax.plot(
                [lp[0], sp[0]],
                [lp[1], sp[1]],
                color="green",
                linewidth=2,
                marker="o",
                linestyle="--",
            )

        ax.set_aspect("equal", "box")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True)
        plt.show()


def linear_interp(qa, qb, eta=0.1):
    dist = np.linalg.norm(qb - qa)
    num_segments = int(np.ceil(dist / eta))
    path = []
    for i in range(num_segments + 1):
        alpha = i / num_segments
        q = (1 - alpha) * qa + alpha * qb
        path.append(q)
    path = np.array(path)
    return path


if __name__ == "__main__":
    shapes = {
        # "shape1": {"x": -0.7, "y": 1.3, "h": 2, "w": 2.2},
        "shape1": {"x": -0.7, "y": 2.1, "h": 2, "w": 2.2},
        "shape2": {"x": 2, "y": -2.0, "h": 1, "w": 4.0},
        "shape3": {"x": -3, "y": -3, "h": 1.25, "w": 2},
    }
    obstacles = [
        box(k["x"], k["y"], k["x"] + k["w"], k["y"] + k["h"])
        for k in shapes.values()
    ]
    robot = PlanarRR()
    scene = RobotScene(robot, obstacles)

    # scene.cspace_obstacles(generate=True, save=True, plot=False)
    # scene.cspace_dataset_collision()
    # scene.cspace_dataset_nearest_distance()

    theta = np.array([[-np.pi], [np.pi / 4.0]])
    theta = np.array([[np.pi / 6.0], [-1.0]])
    best, results = scene.distance_to_obstacles(theta)
    print("Best distance:", best)
    for res in results:
        print(res)
