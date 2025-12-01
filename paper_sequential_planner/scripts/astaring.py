import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import math
import heapq


class MatrixAStar:
    def __init__(self, C=None, config=None, grid_shape=None):
        self.C = C
        self.N = C.shape[0]
        # self.N = config.shape[0]

        self.config = None if config is None else np.asarray(config, dtype=float)
        self.grid_shape = grid_shape

        # precompute row-major strides for index<->coord conversions
        strides = []
        prod = 1
        for s in reversed(self.grid_shape):
            strides.insert(0, prod)
            prod *= s
        self._strides = tuple(strides)

        # precompute neighbor adjacency for every node (N x neighbors)
        self._neighbors = [[] for _ in range(self.N)]
        for idx in range(self.N):
            coord = self.index_to_coordinate(idx)
            neighs = []
            for dim in range(len(self.grid_shape)):
                for delta in (-1, 1):
                    new_coord = list(coord)
                    new_coord[dim] += delta
                    if 0 <= new_coord[dim] < self.grid_shape[dim]:
                        new_idx = int(
                            sum(
                                int(c) * s
                                for c, s in zip(new_coord, self._strides)
                            )
                        )
                        neighs.append(new_idx)
            self._neighbors[idx] = neighs

    def heuristic(self, i, j):
        return float(np.linalg.norm(self.config[i] - self.config[j]))

    def distance_between(self, i, j):
        if self.config is not None:
            return float(np.linalg.norm(self.config[i] - self.config[j]))
        if self.C is None:
            raise ValueError(
                "No cost source available: provide `config` or dense `C`"
            )
        return float(self.C[i, j])

    def neighbors(self, idx):
        # Return precomputed neighbors for the grid. This is O(1) per call.
        if self._neighbors is None:
            raise ValueError(
                "Neighbors not initialized; grid_shape must be provided"
            )
        return self._neighbors[idx]

    def coordinate_to_index(self, coord):
        """Convert an integer coordinate tuple to a node index using row-major ordering.

        For legacy 2D ordering used in this script the mapping is x*height + y.
        For N-d `grid_shape` the mapping uses precomputed strides.
        """
        if len(coord) != len(self.grid_shape):
            raise ValueError("coordinate length does not match grid_shape")
        return int(sum(int(c) * s for c, s in zip(coord, self._strides)))

    def index_to_coordinate(self, idx):
        """Convert a node index to an integer coordinate tuple (row-major)."""
        rem = int(idx)
        coords = []
        for s in self._strides:
            v = rem // s
            coords.append(int(v))
            rem = rem % s
        return tuple(coords)

    def astar(self, start, goal):
        """Run A* from start to goal. Returns list of node indices (empty if not reachable)."""
        # allow passing coordinates (tuples) as start/goal; convert to indices if needed
        start_idx = (
            self.coordinate_to_index(start)
            if isinstance(start, (tuple, list))
            else start
        )
        goal_idx = (
            self.coordinate_to_index(goal)
            if isinstance(goal, (tuple, list))
            else goal
        )

        if start_idx == goal_idx:
            return [start_idx]

        open_heap = []  # (f_score, node)
        g_score = {i: math.inf for i in range(self.N)}
        came_from = {}

        g_score[start_idx] = 0.0
        start_f = g_score[start_idx] + self.heuristic(start_idx, goal_idx)
        heapq.heappush(open_heap, (start_f, start_idx))
        open_set = {start_idx}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_idx:
                return self._reconstruct_path(came_from, current)

            open_set.discard(current)

            for neighbor in self.neighbors(current):
                tentative_g = g_score[current] + self.distance_between(
                    current, neighbor
                )
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal_idx)
                    if neighbor not in open_set:
                        heapq.heappush(open_heap, (f, neighbor))
                        open_set.add(neighbor)

        # not found
        return []

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _path_convert_to_coordinates(self, path):
        return [self.index_to_coordinate(i) for i in path]


class NDimGridAStar:

    def __init__(self, costgrid):
        self.costgrid = costgrid
        self.costtrack = np.full_like(costgrid, np.inf, dtype=float)
        self.visitednode = []
        self.dof = len(costgrid.shape)
        self.grid_shape = costgrid.shape
        self.N = costgrid.size

    def __str__(self):
        return (
            f"NDim-AStar,"
            + f"\n-dof={self.dof},"
            + f"\n-grid_shape={self.grid_shape}"
        )

    def distance_between(self, u, v):
        """
        compute distance between two nodes u and v
        u and v are tuples of coordinates
        ex: u = (x1, y1, z1)
            v = (x2, y2, z2)
        """
        return float(np.linalg.norm(np.array(u) - np.array(v)))

    def heuristic(self, u, v):
        """
        compute heuristic between two nodes u and v
        u and v are tuples of coordinates
        ex: u = (x1, y1, z1)
            v = (x2, y2, z2)
        """
        return float(np.linalg.norm(np.array(u) - np.array(v)))

    def neighbors(self, nodeid):
        """fix lower and upper check later incase of non-square grid"""
        motion = self.motions()
        nn = nodeid + motion
        l = nn >= 0
        u = nn <= self.grid_shape[0]
        valid_mask = np.all(l & u, axis=1)
        valid_nn = nn[valid_mask]
        return valid_nn

    def motions(self):
        motion = np.array(list(product([-1, 0, 1], repeat=self.dof)))
        return motion

    def astar(self, start, goal):
        if start == goal:
            return [start]

        open_heap = []  # (f_score, node)
        # case of not found
        return []


# class AStar:

#     def __init__(self, graph) -> None:
#         self.graph = graph
#         self.heapq = graph
#         self.visitedNode = []

#     def heapq_prio_heuristic(self, goal: Node):  # there are some method better than this but i want to test myself
#         costs = np.array([hg.cost for hg in self.heapq])
#         coststogo = np.array([self.cost_to_go(goal, hg) for hg in self.heapq])
#         ci = np.argsort(costs + coststogo)
#         self.heapq = [self.heapq[i] for i in ci]

#     def cost_to_go(self, xTo, xFrom):  # euclidean distance
#         return np.linalg.norm(xTo.config - xFrom.config)

#     def backtrack(self, node: Node):
#         path = [node]
#         current = node
#         while current.pathvia is not None:
#             path.append(current.pathvia)
#             current = current.pathvia
#         return path

#     def search(self, start: Node, goal: Node):
#         # set start at 0
#         start.cost = 0

#         while True:
#             if len(self.visitedNode) == len(self.graph):
#                 print("no path were found")
#                 return

#             self.heapq_prio_heuristic(goal)
#             currentNode = self.heapq[0]

#             if currentNode is goal:
#                 return self.backtrack(currentNode)

#             for ei, ed in enumerate(currentNode.edgeNodes):
#                 if ed in self.visitedNode:
#                     continue
#                 if (pcost := currentNode.cost + currentNode.edgeCosts[ei]) < ed.cost:
#                     ed.cost = pcost
#                     ed.pathvia = currentNode
#             self.visitedNode.append(currentNode)
#             self.heapq.pop(0)


if __name__ == "__main__":
    from geometric_pcm import make_costgrid, make_geometric_grid
    from itertools import product

    costgrid = make_costgrid(npoints=10, dof=2)
    print("shape:", costgrid.shape)
    sqrcenter, length = make_geometric_grid(npoints=10, dof=2)
    print("sqrcenter:\n", sqrcenter)

    start_index = (0, 0)
    end_index = (6, 6)

    astar_planner = NDimGridAStar(costgrid)
    print(astar_planner)

    node = np.array([0, 0])
    nn = astar_planner.neighbors(node)
    print("nn:\n", nn)

    # plt.imshow(costgrid, cmap="gray")
    # plt.show()
