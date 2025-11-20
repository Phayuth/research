from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import numpy as np
import time

# ou.RNG.setSeed(int(time.time() * 1000000) % 2**32)


def planWithSimpleSetup():

    def isStateValid(state):
        # Some arbitrary condition on the state (note that thanks to
        # dynamic type checking we can just call getX() and do not need
        # to convert state to an SE2State.)
        q = [state[0], state[1], state[2], state[3], state[4], state[5]]
        return True

    space = ob.RealVectorStateSpace(6)
    bounds = ob.RealVectorBounds(6)
    limit6 = [2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi]
    for i in range(6):
        bounds.setLow(i, -limit6[i])
        bounds.setHigh(i, limit6[i])
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    start = ob.State(space)
    start[0] = 0.0
    start[1] = 0.0
    start[2] = 0.0
    start[3] = 0.0
    start[4] = 0.0
    start[5] = 0.0
    goal = ob.State(space)
    goal[0] = 1.0
    goal[1] = 1.0
    goal[2] = 0.0
    goal[3] = 0.0
    goal[4] = 0.0
    goal[5] = 0.0

    # start.random()
    # goal.random()
    ss.setStartAndGoalStates(start, goal)

    ss.setPlanner(og.RRTConnect(ss.getSpaceInformation()))
    solved = ss.solve(1.0)

    if solved:
        ss.simplifySolution()
        path = ss.getSolutionPath()

        pathlist = []
        for i in range(path.getStateCount()):
            pi = path.getState(i)
            pathlist.append([pi[0], pi[1], pi[2], pi[3], pi[4], pi[5]])
        print("Found solution:")
        print(pathlist)
        return pathlist
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
    from plot_ur5e_bullet import UR5eBullet, Constants

    robot = UR5eBullet("no_gui")
    model_id = robot.load_models_other(Constants.model_list_shelf)
    # 4.439302803306453
    planner = OMPLPlanner(robot.collision_check_at_config)
    qs = np.array([5.177, -0.957, 0.726, 0.231, 2.036, -3.142])
    qg = np.array([5.177, -0.56, 1.384, 2.318, 4.248, -6.283])
    result = planner.query_planning(qs, qg)
    if result is not None:
        pathlist, path_cost = result
        print("Path list:")
        print(pathlist)
        print(f"Path cost: {path_cost}")
