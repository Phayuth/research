import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box, Polygon, MultiPolygon
from shapely.ops import nearest_points
from scipy.optimize import minimize

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

    def ik(self, X, q_init):
        X = np.asarray(X, dtype=float).reshape(
            2,
        )
        q_init = np.asarray(q_init, dtype=float).reshape(
            6,
        )

        def objective(q):
            X_pred = self.fk_vectorized(q[np.newaxis, :])[0][-1]
            e = X_pred - X
            return float(np.dot(e, e))

        res = minimize(
            objective,
            q_init,
            method="L-BFGS-B",
            bounds=[(-2 * np.pi, 2 * np.pi)] * 6,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not res.success:
            return None
        return res.x


class RobotScene:

    def __init__(self, robot, obstacles):
        self.robot = robot
        o1 = Polygon([(-2, -4), (3, -4), (3, -2.5), (-3, -2.5)])
        o2 = Polygon([(-4, 3), (0, 3), (0, 4), (-4, 4)])
        o3 = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
        self.obstacles = MultiPolygon([o1, o2, o3])
        # o1 = Polygon([(-6, -6), (-2, -6), (-2, 6), (-6, 6)])
        # self.obstacles = MultiPolygon([o1])
        self.arm_thinkness = 0.2

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


class OMPLPlanner:

    def __init__(self, collision_checker):
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
        col = self.collision_checker(q)
        return not col

    def query_planning(self, start_list, goal_list):
        # Important!
        # Clear previous planning data to ensure fresh planning because caching
        self.ss.clear()

        start = ob.State(self.space)
        start[0] = start_list[0]
        start[1] = start_list[1]
        start[2] = start_list[2]
        start[3] = start_list[3]
        start[4] = start_list[4]
        start[5] = start_list[5]
        goal = ob.State(self.space)
        goal[0] = goal_list[0]
        goal[1] = goal_list[1]
        goal[2] = goal_list[2]
        goal[3] = goal_list[3]
        goal[4] = goal_list[4]
        goal[5] = goal_list[5]

        dist = np.linalg.norm(np.array(goal_list) - np.array(start_list))

        self.ss.setStartAndGoalStates(start, goal)
        status = self.ss.solve(100.0)
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


if __name__ == "__main__":
    robot = Planar6R()
    scene = RobotScene(robot, None)
    q = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
    scene.show_env(q)

    xd = np.array([4.0, 2.0])
    qd = robot.ik(xd, q)
    print("qd:", qd)
    scene.show_env(qd)
