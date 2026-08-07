import gurobipy as gp
from gurobipy import GRB


def parse_gtsp(filename):
    nodes = []
    clusters = []
    cost = None
    dimension = None

    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):  # go through each line of the file
        line = lines[i].strip()

        # Read dimension
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1])

        # Read distance matrix
        elif line == "EDGE_WEIGHT_SECTION":
            matrix = []
            i += 1
            while len(matrix) < dimension * dimension:
                values = lines[i].strip().split()
                if values:
                    matrix.extend([float(x) for x in values])
                i += 1
            cost = np.array(matrix).reshape(dimension, dimension)
            continue

        # Read clusters
        elif line == "GTSP_SET_SECTION":
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line == "EOF":
                    break
                values = [int(x) for x in line.split()]
                clustnodes = values[1:]  # remove cluster index
                clustnodes = [x - 1 for x in clustnodes if x != -1]  # remove -1
                clusters.append(clustnodes)
                i += 1
            break
        i += 1
    nodes = list(range(dimension))
    return nodes, clusters, cost


def solve_gtsp_mip(nodes, clusters, cost):
    # all possible edges
    edges = [(i, j) for i in nodes for j in nodes if i != j]

    # Subtour detector
    def find_subtour(selected_edges):
        neighbors = {}
        for i, j in selected_edges:
            neighbors.setdefault(i, [])
            neighbors[i].append(j)
        cycles = []
        visited = set()
        for start in neighbors:
            if start in visited:
                continue
            cycle = []
            current = start
            while current not in cycle:
                cycle.append(current)
                visited.add(current)
                current = neighbors[current][0]
            cycles.append(cycle)

        return cycles

    # Lazy constraint callback
    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_values = model.cbGetSolution(x_vars)
            selected = [(i, j) for i, j in edges if x_values[i, j] > 0.5]
            cycles = find_subtour(selected)

            # if more than one cycle, cut them
            if len(cycles) > 1:
                for cycle in cycles:
                    model.cbLazy(
                        gp.quicksum(
                            x_vars[i, j] for i in cycle for j in cycle if i != j
                        )
                        <= len(cycle) - 1
                    )

    # Build GTSP model
    model = gp.Model("GTSP")
    y = model.addVars(
        nodes, vtype=GRB.BINARY, name="select"
    )  # node selected variable
    x = model.addVars(edges, vtype=GRB.BINARY, name="edge")  # edge variable
    x_vars = x

    # Objective
    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for i, j in edges), GRB.MINIMIZE
    )

    # One node from each cluster
    for cluster in clusters:
        model.addConstr(gp.quicksum(y[i] for i in cluster) == 1)

    # Flow constraints
    for i in nodes:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == y[i])
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if j != i) == y[i])

    # allow lazy cuts
    model.Params.LazyConstraints = 1

    # Set parameters
    model.Params.MemLimit = 15  # GB
    model.Params.Threads = 4

    # Solve
    model.optimize(subtour_callback)

    # Extract solution
    if model.status == GRB.OPTIMAL:
        print("Guaranteed optimal")
    else:
        print("Not proven optimal")

    selected_edges = [(i, j) for i, j in edges if x[i, j].X > 0.5]
    selected_nodes = [i for i in nodes if y[i].X > 0.5]
    print("\nOptimal cost:")
    print(model.ObjVal)
    print("\nSelected IK nodes:")
    print(selected_nodes)
    print("\nTour edges:")
    print(selected_edges)


if __name__ == "__main__":
    # Small GTSP example
    # clusters = [[0, 1], [2, 3], [4, 5]]
    # nodes = list(range(6))
    # cost = [
    #     [0, 4, 8, 6, 7, 3],
    #     [4, 0, 5, 9, 2, 8],
    #     [8, 5, 0, 3, 6, 4],
    #     [6, 9, 3, 0, 5, 7],
    #     [7, 2, 6, 5, 0, 4],
    #     [3, 8, 4, 7, 4, 0],
    # ]
    nodes, clusters, cost = parse_gtsp(
        "./paper_sequential_planner/theorems/single_stool_hGTSP_mjd_minits.gtsp"
    )
    solve_gtsp_mip(nodes, clusters, cost)
