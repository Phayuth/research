from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import numpy as np
import time

ou.RNG.setSeed(int(time.time() * 1000000) % 2**32)


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
        status = self.ss.solve(1.0)
        print("-----------")
        print(dir(status))
        print("EXACT") if status.EXACT_SOLUTION else print("INEXACT")
        print("-----------")
        if status:
            self.ss.simplifySolution()
            path = self.ss.getSolutionPath()
            print("Found solution:")
            print(self.ss.getSolutionPath())

            pathlist = []
            for i in range(path.getStateCount()):
                pi = path.getState(i)
                pathlist.append([pi[0], pi[1], pi[2], pi[3], pi[4], pi[5]])
            return pathlist
        else:
            print("No solution found")


if __name__ == "__main__":
    from data_compute_scene import UR5eBullet

    planner = OMPLPlanner()
