from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import numpy as np
import time

ou.RNG.setSeed(int(4.2 * 1000000) % 2**32)


class OMPLSO2Planner:

    def __init__(self, collision_checker):
        self.collision_checker = collision_checker

        self.space = ob.CompoundStateSpace()
        self.space.addSubspace(ob.SO2StateSpace(), 1.0)  # Joint 1
        self.space.addSubspace(ob.SO2StateSpace(), 1.0)  # Joint 2
        self.space.addSubspace(ob.SO2StateSpace(), 1.0)  # Joint 3
        self.space.addSubspace(ob.SO2StateSpace(), 1.0)  # Joint 4
        self.space.addSubspace(ob.SO2StateSpace(), 1.0)  # Joint 5
        self.space.addSubspace(ob.SO2StateSpace(), 1.0)  # Joint 6

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )
        self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        self.planner.setRange(0.1)
        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        q = [
            state[0].value,
            state[1].value,
            state[2].value,
            state[3].value,
            state[4].value,
            state[5].value,
        ]
        print(q)
        col = self.collision_checker(q)
        return not col

    def query_planning(self, start_list, goal_list):
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

        self.ss.setStartAndGoalStates(start, goal)
        status = self.ss.solve(10.0)
        print("EXACT") if status.EXACT_SOLUTION else print("INEXACT")
        if status:
            # self.ss.simplifySolution()
            path = self.ss.getSolutionPath()
            # Option 1: Use path length (simpler)
            path_cost = path.length()

            print("Found solution:")
            print(f"Path cost: {path_cost}")
            print(self.ss.getSolutionPath())

            pathlist = []
            for i in range(path.getStateCount()):
                pi = path.getState(i)
                pathlist.append(
                    [
                        pi[0].value,
                        pi[1].value,
                        pi[2].value,
                        pi[3].value,
                        pi[4].value,
                        pi[5].value,
                    ]
                )
            return pathlist, path_cost
        else:
            print("No solution found")
            return None


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
        self.space.setBounds(self.bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )
        self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        self.planner.setRange(0.1)
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

        self.ss.setStartAndGoalStates(start, goal)
        status = self.ss.solve(10.0)
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


def compute_path_cost(path):
    cost = 0.0
    for i in range(1, len(path)):
        cost += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
    return cost


if __name__ == "__main__":
    import pandas as pd
    from data_compute_scene import UR5eBullet, Constants
    from spatial_geometry.utils import Utils

    def so2planning():
        robot = UR5eBullet("no_gui")
        model_id = robot.load_models_other(Constants.model_box_strong_obstacle)

        planner = OMPLSO2Planner(robot.collision_check_at_config)
        qs = np.array(
            [
                0.39683294296264565,
                -0.9920818805694598,
                1.5873310565948522,
                -0.8928737640380842,
                1.8518865108490026,
                -3.141592741012566,
            ]
        )
        qg = np.array(
            [
                -3.6376335620880056,
                -1.7857475280761683,
                -1.7196087837219274,
                -2.1825799942016664,
                -1.9180250167846644,
                -0.2645549774169931,
            ]
        )
        qs = Utils.wrap_to_pi(qs)
        qg = Utils.wrap_to_pi(qg)
        path, cost = planner.query_planning(qs, qg)

        # save path to csv
        df_path = pd.DataFrame(path, columns=[f"q{i}" for i in range(6)])
        df_path.to_csv("data_so2_path.csv", index=False)

    # so2planning()

    def euclidean_planning():
        robot = UR5eBullet("no_gui")
        model_id = robot.load_models_other(Constants.model_box_strong_obstacle)

        planner = OMPLPlanner(robot.collision_check_at_config)
        Qalt = pd.read_csv("data_nik_qalt.csv").to_numpy()
        qs = np.array(
            [
                0.39683294296264565,
                -0.9920818805694598,
                1.5873310565948522,
                -0.8928737640380842,
                1.8518865108490026,
                -3.141592741012566,
            ]
        )

        cost = []
        for i in range(Qalt.shape[0]):
            print(f"Planning to goal {i}")
            qg = Qalt[i]
            print(f"qg: {qg}")
            result = planner.query_planning(qs, qg)
            if result is not None:
                path, path_cost = result
                cost.append(path_cost)
                # cost_manual = compute_path_cost(path)
                print(f"Path cost for goal {i}: {path_cost}")
                # print(f"Manual computed path cost for goal {i}: {cost_manual}")
                # save path to csv per file
                df_path = pd.DataFrame(path, columns=[f"q{i}" for i in range(6)])
                df_path.to_csv(
                    f"./paths/data_euclidean_path_goal_{i}.csv", index=False
                )
            else:
                print(f"No solution found for goal {i}")
                cost.append(float("inf"))  # or None, depending on your preference

        # concat Qalt and cost and save to csv
        df_cost = pd.DataFrame(cost, columns=["cost"])
        df_qalt = pd.DataFrame(Qalt, columns=[f"q{i}" for i in range(6)])
        df_result = pd.concat([df_qalt, df_cost], axis=1)
        df_result.to_csv("data_planner_results.csv", index=False)

    # euclidean_planning()
    def exp1_planning():
        robot = UR5eBullet("no_gui")
        # model_id = robot.load_models_other(Constants.model_list)

        planner = OMPLPlanner(robot.collision_check_at_config)

        exp_param = {
            "qsinpi_direct": [
                "data_ur5e_qs_inpi.csv",
                "data_ur5e_Qik.csv",
            ],
            "qsoutpi_direct": [
                "data_ur5e_qs_outpi.csv",
                "data_ur5e_Qik.csv",
            ],
            "qsinpi_alt": [
                "data_ur5e_qs_inpi.csv",
                "data_ur5e_qemins_inpi.csv",
            ],
            "qsoutpi_alt": [
                "data_ur5e_qs_outpi.csv",
                "data_ur5e_qemins_outpi.csv",
            ],
        }

        exp = "qsoutpi_alt"

        qs = pd.read_csv(exp_param[exp][0]).to_numpy().flatten()
        Qik = pd.read_csv(exp_param[exp][1]).to_numpy()

        cost = []
        for i in range(Qik.shape[0]):
            print(f"Planning to goal {i}")
            qg = Qik[i]
            print(f"qg: {qg}")
            result = planner.query_planning(qs, qg)
            if result is not None:
                path, path_cost = result
                print(f"Path cost for goal {i}: {path_cost}")
                cost.append(path_cost)
                df_path = pd.DataFrame(path, columns=[f"q{i}" for i in range(6)])
                df_path.to_csv(
                    f"./paths_exp1/data_exp1_path_goal_{i}.csv", index=False
                )
            else:
                print(f"No solution found for goal {i}")
                cost.append(float("inf"))  # or None, depending on your preference
        # concat Qik and cost and save to csv
        df_cost = pd.DataFrame(cost, columns=["cost"])
        df_Qik = pd.DataFrame(Qik, columns=[f"q{i}" for i in range(6)])
        df_result = pd.concat([df_Qik, df_cost], axis=1)
        df_result.to_csv(f"data_exp1_planner_results_{exp}.csv", index=False)

    exp1_planning()
