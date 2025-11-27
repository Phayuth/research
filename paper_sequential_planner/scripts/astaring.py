import heapq
import numpy as np
import math


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


if __name__ == "__main__":
    # build a grid of configuration points
    nodex = np.linspace(-np.pi, np.pi, 9)
    nodey = np.linspace(-np.pi, np.pi, 9)
    nx = len(nodex)
    ny = len(nodey)
    config = np.array([[x, y] for x in nodex for y in nodey])

    # build cost matrix but we only need costs between neighbors — full matrix is fine
    C = np.zeros((config.shape[0], config.shape[0]))
    for i in range(config.shape[0]):
        for j in range(config.shape[0]):
            if i != j:
                dist = np.linalg.norm(config[i] - config[j])
                C[i][j] = dist
            else:
                C[i][j] = 0.0

    solver = MatrixAStar(C, config=config, grid_shape=(nx, ny))
    sid = (0, 0)
    eid = (6, 6)
    path = solver.astar(sid, eid)
    path_coords = solver._path_convert_to_coordinates(path)
    print("path (indices):", path)
    print("path (coords):", path_coords)

    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.colors import LinearSegmentedColormap

    vline = np.linspace(-np.pi, np.pi, 10)
    hline = np.linspace(-np.pi, np.pi, 10)
    costgrid = np.random.rand(len(vline) - 1, len(hline) - 1)
    freecolor = "white"
    colcolor = "red"
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [freecolor, colcolor])

    fig, ax = plt.subplots()
    # make grid rectangles patches
    for i in range(len(vline) - 1):
        for j in range(len(hline) - 1):
            rect = patches.Rectangle(
                (vline[i], hline[j]),
                vline[i + 1] - vline[i],
                hline[j + 1] - hline[j],
                linewidth=0.5,
                edgecolor="gray",
                facecolor=cmap(costgrid[i, j]),
            )
            ax.add_patch(rect)
            text = f"({i},{j}) {costgrid[i, j]:.2f}"
            ax.text(
                vline[i] + 0.1,
                hline[j] + 0.1,
                text,
                fontsize=6,
                color="gray",
            )
    # plot path
    for px, py in path_coords:
        rect = patches.Rectangle(
            (vline[px], hline[py]),
            vline[px + 1] - vline[px],
            hline[py + 1] - hline[py],
            linewidth=1.5,
            edgecolor="gray",
            facecolor="blue",
        )
        ax.add_patch(rect)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", "box")
    plt.show()
