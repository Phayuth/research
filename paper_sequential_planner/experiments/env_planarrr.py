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
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class PlanarRR:

    def __init__(self):
        self.a1 = 2
        self.a2 = 2

    def forward_kinematic(self, theta):
        theta = np.asarray(theta)
        if theta.shape == (2,):
            theta = theta.reshape(2, 1)
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

    def collision_checker(self, q):
        best, res = self.distance_to_obstacles(q)
        if best["distance"] <= 0:
            return True
        else:
            return False

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
        Xsample = sample_reachable_wspace(30)

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

        ntasks = 30
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


class OMPLPlanner:

    def __init__(self, collision_checker):
        self.collision_checker = collision_checker
        self.dof = 2
        self.space = ob.RealVectorStateSpace(self.dof)
        self.bounds = ob.RealVectorBounds(self.dof)
        self.limit2 = [
            2 * np.pi,
            2 * np.pi,
        ]
        for i in range(self.dof):
            self.bounds.setLow(i, -self.limit2[i])
            self.bounds.setHigh(i, self.limit2[i])
        self.space.setBounds(self.bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )
        # self.planner = og.BITstar(self.ss.getSpaceInformation())
        # self.planner = og.ABITstar(self.ss.getSpaceInformation())
        # self.planner = og.AITstar(self.ss.getSpaceInformation())
        self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        self.planner.setRange(0.1)
        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        q = [state[0], state[1]]
        col = self.collision_checker(q)
        return not col

    def query_planning(self, start_list, goal_list):
        # Important!
        # Clear previous planning data to ensure fresh planning because caching
        self.ss.clear()

        start = ob.State(self.space)
        start[0] = start_list[0]
        start[1] = start_list[1]
        goal = ob.State(self.space)
        goal[0] = goal_list[0]
        goal[1] = goal_list[1]

        lowest = np.linalg.norm(np.array(goal_list) - np.array(start_list))

        self.ss.setStartAndGoalStates(start, goal)
        status = self.ss.solve(5.0)
        print(
            "Plan from ", start_list, " to ", goal_list, "estimate cost:", lowest
        )
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
                pathlist.append([pi[0], pi[1]])
            return pathlist, path_cost
        else:
            print("No solution found")
            return None


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
    limit = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
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
    from paper_sequential_planner.scripts.rtsp_lazyprm import (
        separate_sample,
        build_graph,
        estimate_shortest_path,
    )

    robot = PlanarRR()
    scene = RobotScene(robot, None)
    planner = OMPLPlanner(scene.collision_checker)
    # scene.cspace_obstacles(generate=True, save=True, plot=False)
    # scene.cspace_dataset_collision()
    # scene.cspace_dataset_nearest_distance()

    q = np.array([[np.pi / 6.0], [-1.0]])
    # scene.show_env(q)
    scene._show_wsenv_debug(q)
    scene._show_cspace_debug()

    ntasks = 30
    X = sample_reachable_wspace(ntasks)
    MQaik = wspace_ik(robot, X)
    MQaik_validity = wspace_ik_validity(MQaik, scene)
    nsol_per_cluster, MQaik_valid_sols = process_cluster(MQaik, MQaik_validity)
    print(f"==>> nsol_per_cluster: \n{nsol_per_cluster}")
    num_valid_sols = MQaik_valid_sols.shape[0]
    cluster = RTSP.build_cluster(nsol_per_cluster)
    print(f"==>> cluster: \n{cluster}")
    adjm = RTSP.make_adj_matrix(cluster, MQaik_valid_sols.shape[0])
    print(f"==>> adjm: \n{adjm}")
    num_unique_edges = RTSP.find_numedges_unique(nsol_per_cluster)

    QfulRndfree, QfulRndcoll = separate_sample(scene.collision_checker)
    graph, kdtree = build_graph(QfulRndfree, k=10, dist_thres=0.5)

    # adjm_cost_min = adjm.copy()
    # for i in range(adjm_cost_min.shape[0]):
    #     for j in range(adjm_cost_min.shape[1]):
    #         if adjm[i, j] == 1:
    #             q1 = MQaik_valid_sols[i]
    #             q2 = MQaik_valid_sols[j]
    #             adjm_cost_min[i, j] = np.linalg.norm(q2 - q1)
    # print(f"==>> adjm_cost_min: \n{adjm_cost_min}")

    # adjm_cost_est = adjm.copy()
    # for i in range(adjm_cost_est.shape[0]):
    #     for j in range(adjm_cost_est.shape[1]):
    #         print(f"Estimating cost from {i} to {j}...")
    #         if adjm[i, j] == 1:
    #             q1 = MQaik_valid_sols[i]
    #             q2 = MQaik_valid_sols[j]
    #             pathq, cost = estimate_shortest_path(
    #                 q1, q2, QfulRndfree, graph, kdtree
    #             )
    #             print(f"Estimated cost from {i} to {j}: {cost}")
    #             adjm_cost_est[i, j] = cost
    # print(f"==>> adjm_cost_est: \n{adjm_cost_est}")

    # GLKHHelper.write_glkh_fullmatrix_file(
    #     os.path.join(GLKHHelper.problemdir, "problem_planarrr.gtsp"),
    #     adjm_cost_est,
    #     cluster,
    # )

    # solve GTSP using GLKH
    if os.path.exists(
        os.path.join(GLKHHelper.problemdir, "problem_planarrr.tour")
    ):
        tourmatix = GLKHHelper.read_tour_file(
            os.path.join(GLKHHelper.problemdir, "problem_planarrr.tour")
        )
        print(f"==>> tourmatix: \n{tourmatix}")
        cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))

        qtour = MQaik_valid_sols[tourmatix]

        fig, ax = plt.subplots()
        ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
        ax.plot(qtour[:, 0], qtour[:, 1], "go--", markersize=4, label="GTSP tour")
        for i in range(len(tourmatix) - 1):
            start_idx = tourmatix[i]
            end_idx = tourmatix[i + 1]
            q1 = MQaik_valid_sols[start_idx]
            q2 = MQaik_valid_sols[end_idx]
            path = planner.query_planning(q1, q2)
            if path is not None:
                qp, cp = path
                qp = np.array(qp)
                ax.plot(
                    qp[:, 0],
                    qp[:, 1],
                    "b-",
                    alpha=0.5,
                    label="OMPL path" if i == 0 else None,
                )
        ax.set_aspect("equal", "box")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True)
        ax.legend()
        plt.show()
    else:
        print("Tour file not found. Please run GLKH solver file.")
