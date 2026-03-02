import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString, box
from shapely.ops import nearest_points

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    ou.RNG.setSeed(42)

except ImportError:
    print("OMPL not available, limitted functionality without OMPL.")

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
        if X.shape == (2,):
            X = X.reshape(2, 1)
        x = X[0, 0]
        y = X[1, 0]

        # check if the desired pose is inside of task space or not
        rd = np.sqrt(x**2 + y**2)
        link_length = self.a1 + self.a2
        if rd > link_length:
            # print("The desired pose is outside of taskspace")
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

        Q = np.array(
            [
                [theta1_down, theta2_down],
                [theta1_up, theta2_up],
            ]
        )
        return Q


class RobotScene:

    def __init__(self, robot, obstacles):
        self.robot = robot
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
            # print("Generating C-space obstacle plot...")
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
        # print("Generating C-space dataset with collision labels...")
        num_samples = 360
        theta1_samples = np.linspace(-np.pi, np.pi, num_samples)
        theta2_samples = np.linspace(-np.pi, np.pi, num_samples)
        Q1, Q2 = np.meshgrid(theta1_samples, theta2_samples, indexing="ij")
        joints = np.column_stack([Q1.ravel(), Q2.ravel()])
        # print("Generated joint samples:", joints.shape)
        dataset = []
        for i in tqdm.tqdm(range(joints.shape[0])):
            theta = joints[i].reshape(2, 1)
            best, _ = self.distance_to_obstacles(theta)
            if best is not None and best["distance"] <= 0.0:
                dataset.append((joints[i][0], joints[i][1], 1))
            else:
                dataset.append((joints[i][0], joints[i][1], -1))
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

    def show_env(self, theta):
        links_xy = self.robot_collision_links(theta)
        best, results = self.distance_to_obstacles(theta)

        fig, ax = plt.subplots()

        # arm links
        for link in links_xy:
            x, y = link.xy
            ax.plot(x, y, color="blue", linewidth=4, solid_capstyle="round")

        # obstacles
        for shp in self.obstacles:
            x, y = shp.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="red", ec="black")

        # nearest points and distance line
        if best is not None:
            lp = best["link_point"]
            sp = best["shape_point"]
            ax.plot(
                [lp[0], sp[0]],
                [lp[1], sp[1]],
                "o--",
                color="purple",
                markersize=12,
            )
        ax.set_aspect("equal", "box")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True)
        plt.show()

    def _show_wsenv_debug(self, theta):
        links_xy = self.robot_collision_links(theta)
        best, results = self.distance_to_obstacles(theta)
        Xsample = sample_reachable_wspace(100)

        fig, ax = plt.subplots()

        # arm links
        for link in links_xy:
            x, y = link.xy
            ax.plot(x, y, color="blue", linewidth=4, solid_capstyle="round")

        # obstacles
        for shp in self.obstacles:
            x, y = shp.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="red", ec="black")

        # task space samples
        ax.plot(
            Xsample[:, 0],
            Xsample[:, 1],
            "go",
            markersize=4,
            label="Task space samples",
        )

        # nearest points and distance line
        if best is not None:
            lp = best["link_point"]
            sp = best["shape_point"]
            ax.plot(
                [lp[0], sp[0]],
                [lp[1], sp[1]],
                "o--",
                color="purple",
                markersize=12,
            )
        ax.set_aspect("equal", "box")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True)
        plt.show()

    def _show_cspace_debug(self):
        cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))

        ntasks = 100
        X = sample_reachable_wspace(ntasks)
        MQaik = wspace_ik(robot, X)
        MQaik_validity = wspace_ik_validity(MQaik, scene)
        nsol_per_cluster, MQaik_valid_sols = process_cluster(MQaik, MQaik_validity)

        # color code
        hues = np.linspace(0, 1, ntasks, endpoint=False)
        colors = [mcolors.hsv_to_rgb((h, 0.75, 0.9)) for h in hues]

        fig, ax = plt.subplots()
        ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)

        for i in range(ntasks):
            q_sols = MQaik[i]
            ax.plot(
                q_sols[:, 0],
                q_sols[:, 1],
                "o--",
                color=colors[i],
                markersize=8,
                alpha=0.8,
            )

        ax.set_aspect("equal", "box")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        plt.show()


def sample_reachable_wspace(num_points):
    u = np.random.normal(0.0, 1.0, (2 + 2, num_points))
    norms = np.linalg.norm(u, axis=0)
    u = u / norms
    scale = 4 * u[:2, :]  # The first N coordinates are uniform in a unit N ball
    return scale.T


def wspace_ik(robot: PlanarRR, Xtspace):
    """
    Compute AIK of all task points no matter whether they are reachable or not.
    NaN values will be used to indicate unreachable points.
    """
    if Xtspace.shape[1] != 2:
        Xtspace = Xtspace.T
    ntasks = Xtspace.shape[0]
    MQaik = np.full((ntasks, 2, 2), np.nan)  # (ntasks, num_solutions, dof)
    for taski in range(ntasks):
        q_sols = robot.inverse_kinematic(Xtspace[taski])
        if q_sols is not None:
            MQaik[taski] = q_sols
    return MQaik


def wspace_ik_validity(MQaik, robscene: RobotScene):
    """
    Compute the validity of each AIK solution in MQaik.
    1 = Valid
    -1 = NaN (no solution or unreachable)
    -2 = In collision
    -3 = awkward configuration (e.g. near singularity or joint limits)
    """
    # (ntasks, num_solutions)
    MQaik_validity = np.full(shape=(MQaik.shape[0], 2), fill_value=np.nan)
    for taski in range(MQaik.shape[0]):
        for solj in range(MQaik.shape[1]):
            q = MQaik[taski, solj]
            if np.isnan(q).any():
                MQaik_validity[taski, solj] = -1  # No solution
            else:
                best, _ = robscene.distance_to_obstacles(q.reshape(2, 1))
                if best is not None and best["distance"] <= 0.0:
                    MQaik_validity[taski, solj] = -2  # In collision
                else:
                    MQaik_validity[taski, solj] = 1  # Valid
    return MQaik_validity


def process_cluster(MQaik, MQaik_validity):
    # flatten
    MQaik_flat = MQaik.reshape(-1, 2)  # (ntasks * num_solutions, dof)
    MQaik_validity_flat = MQaik_validity.reshape(-1)  # (ntasks * num_solutions,)
    MQaik_valid_sols = MQaik_flat[MQaik_validity_flat == 1]
    nsol_per_cluster = np.sum(MQaik_validity == 1, axis=1)
    # filter out clusters with zero valid solutions
    nsol_per_cluster_final = nsol_per_cluster[nsol_per_cluster > 0]
    return nsol_per_cluster_final, MQaik_valid_sols


if __name__ == "__main__":
    from paper_sequential_planner.scripts.rtsp_solver import RTSP, GLKHHelper

    robot = PlanarRR()
    scene = RobotScene(robot, None)

    # scene.cspace_obstacles(generate=True, save=True, plot=False)
    # scene.cspace_dataset_collision()
    # scene.cspace_dataset_nearest_distance()

    # q = np.array([[np.pi / 6.0], [-1.0]])
    # scene.show_env(q)
    # scene._show_wsenv_debug(q)
    # scene._show_cspace_debug()

    ntasks = 100
    X = sample_reachable_wspace(ntasks)
    MQaik = wspace_ik(robot, X)
    MQaik_validity = wspace_ik_validity(MQaik, scene)
    nsol_per_cluster, MQaik_valid_sols = process_cluster(MQaik, MQaik_validity)
    cluster = RTSP.build_cluster(nsol_per_cluster)
    print(f"==>> cluster: {cluster}")
    adjm = RTSP.make_adj_matrix(cluster, MQaik_valid_sols.shape[0])
    print(f"==>> adjm: {adjm}")
    num_unique_edges = RTSP.find_numedges_unique(nsol_per_cluster)
    print(f"==>> num_unique_edges: {num_unique_edges}")
