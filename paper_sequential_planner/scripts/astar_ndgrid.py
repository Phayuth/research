import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import heapq


class AStarNDimGrid:

    def __init__(self, costgrid):
        self.costgrid = costgrid
        self.dof = len(costgrid.shape)
        self.grid_shape = costgrid.shape
        self.N = costgrid.size

        # precompute
        self.motion = self.motions()
        self.motion_cost = self.motions_cost(self.motion)

    def __str__(self):
        return (
            f"NDim-AStar,"
            + f"\n-dof={self.dof},"
            + f"\n-grid_shape={self.grid_shape}"
        )

    def heuristic(self, u, v):
        return float(np.linalg.norm(np.array(u) - np.array(v)))

    def neighbors(self, nodeid):
        """fix lower and upper check later incase of non-square grid"""
        # motion = self.motions()
        # motioncost = self.motions_cost(motion)

        nn = nodeid + self.motion
        # indexing start from 0 to size of grid -1 as 0 based indexing
        l = nn >= 0
        u = nn <= self.grid_shape[0] - 1
        valid_mask = np.all(l & u, axis=1)
        valid_nn = nn[valid_mask]
        cost = self.motion_cost[valid_mask]
        return valid_nn, cost

    def motions(self):
        motion = np.array(list(product([-1, 0, 1], repeat=self.dof)))
        zeros = np.all(motion == 0, axis=1)
        motion = motion[~zeros]
        return motion

    def motions_cost(self, motions):
        p = np.prod(motions, axis=1)
        h = p == 0
        g = p != 0
        y = np.zeros_like(p, dtype=float)
        y[h] = 1
        y[g] = np.sqrt(2)
        return y

    def backtrack(self, node):
        print("found path")
        path = [node]
        current = node
        while tuple(current) in self.pathvia:
            current = self.pathvia[tuple(current)]
            path.append(current)
        path.reverse()
        return path

    def solve(self, start, goal):
        if start == goal:
            return [start]

        self.costtrack = np.full_like(self.costgrid, np.inf, dtype=float)
        self.costtrack[start] = 0.0 + self.heuristic(start, goal)
        self.openheap = []
        self.visitednode = []
        self.pathvia = {}

        heapq.heappush(self.openheap, (self.costtrack[start].item(), start))

        while self.openheap:
            ccost, cnode = heapq.heappop(self.openheap)

            # found path if current node is goal
            if cnode == goal:
                return self.backtrack(cnode)

            # get neighbors of current node and score
            nn, nn_movecost = self.neighbors(cnode)

            for ni, n in enumerate(nn):
                # if neighbor already visited, skip
                if tuple(n) in self.visitednode:
                    continue
                # update cost and viapath
                cncost = self.costtrack[tuple(n)]
                movecost = nn_movecost[ni]
                cnheuristic = self.heuristic(n, goal)
                if (pcost := ccost + movecost + cnheuristic) < cncost:
                    self.costtrack[tuple(n)] = pcost
                    self.pathvia[tuple(n)] = tuple(cnode)
                    heapq.heappush(self.openheap, (pcost, tuple(n)))
            self.visitednode.append(cnode)

        # no path found
        return []


class AStarNDimGridProbabilistic(AStarNDimGrid):

    def __init__(self, costgrid):
        super().__init__(costgrid)


def performance_test(npoints=10, dof=2):
    import time
    from geometric_pcm import make_costgrid, make_geometric_grid

    costgrid = make_costgrid(npoints=npoints, dof=dof)
    sqrcenter, length = make_geometric_grid(npoints=npoints, dof=dof)

    start_index = (0,) * dof
    end_index = (4,) * dof

    astar_planner = AStarNDimGrid(costgrid)
    print(astar_planner)

    start_time = time.time()
    path = astar_planner.solve(start_index, end_index)
    end_time = time.time()
    print(f"planning time: {end_time - start_time:.4f} seconds")
    print("final path:", path)

    # for p in path:
    #     costgrid[p] = 0.5  # mark path on costgrid
    # plt.imshow(costgrid, cmap="gray")
    # plt.show()


if __name__ == "__main__":
    performance_test(npoints=10, dof=2)
    performance_test(npoints=10, dof=3)
    performance_test(npoints=10, dof=4)
    performance_test(npoints=10, dof=5)
    performance_test(npoints=10, dof=6)
