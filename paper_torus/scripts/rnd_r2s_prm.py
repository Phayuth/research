import networkx as nx
import numpy as np
import os
from spatial_geometry.utils import Utils
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=2000, precision=2, suppress=True)


class PlotterConfig:
    globalLinewidth = 1

    obstColor = "darkcyan"
    obstFaceColor = "darkcyan"
    obstMarker = "o"
    obstMarkersize = 1.5

    treeColor = "darkgray"
    treeFaceColor = None
    treeMarker = None
    treeMarkersize = None

    stateStartColor = "blue"
    stateStartFaceColor = "yellow"
    stateAppColor = "blue"
    stateAppFaceColor = "green"
    stateGoalColor = "blue"
    stateGoalFaceColor = "red"
    stateMarkersize = 7
    stateMarker = "o"

    pathColor = "blue"
    pathFaceColor = "plum"
    pathMarker = "o"
    pathMarkersize = 7


def solve_graph_mip(graph, startnode=0, endnode=5):
    """networkflow modelling"""
    from pyomo.environ import (
        ConcreteModel,
        Var,
        Objective,
        SolverFactory,
        NonNegativeReals,
        Integers,
        minimize,
        Binary,
        ConstraintList,
    )
    import re

    # this looks exactly like network flow problem solved using LP
    model = ConcreteModel()

    A = []
    w = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            # add any graph edge that is not 0 and tail is not startnode and head is not endnode
            if graph[i][j] != 0 and j != startnode and i != endnode:
                A.append(f"{i}_x_{j}")
                w.append(graph[i][j])

    model.x = Var(A, within=Binary)
    model.obj = Objective(
        expr=sum(w[i] * model.x[e] for i, e in enumerate(A)), sense=minimize
    )
    model.constraints = ConstraintList()

    print(A)
    print(w)

    startpattern = rf"{startnode}_x_.*"
    endpattern = rf"^.*_x_{endnode}$"

    startconstraints = [s for s in A if re.match(startpattern, s)]
    print(f"> startconstraints: {startconstraints}")
    endconstraints = [s for s in A if re.match(endpattern, s)]
    print(f"> endconstraints: {endconstraints}")

    # start and end constraints
    # sum of start constraints must be 1
    # sum of end constraints must be 1
    model.constraints.add(expr=sum(model.x[xi] for xi in startconstraints) == 1)
    model.constraints.add(expr=sum(model.x[xi] for xi in endconstraints) == 1)

    # define the constraints for bidirectional edges
    for i in range(len(graph)):
        if i == startnode or i == endnode:
            continue
        sp = rf"{i}_x_.*"
        ep = rf"^.*_x_{i}$"
        sc = [s for s in A if re.match(sp, s)]
        print(f"> sc: {sc}")
        ec = [s for s in A if re.match(ep, s)]
        print(f"> ec: {ec}")
        if len(sc) == 0 or len(ec) == 0:
            continue
        model.constraints.add(
            expr=sum(model.x[xi] for xi in sc) == sum(model.x[xi] for xi in ec)
        )
        print("add constraint")

    opt = SolverFactory("glpk")
    result_obj = opt.solve(model, tee=True)

    model.pprint()

    opt_solution = [model.x[item].value for item in A]
    print(f"> opt_solution: {opt_solution}")

    xsol = [e for i, e in enumerate(A) if opt_solution[i] == 1]
    xsolv = [i for i, e in enumerate(A) if opt_solution[i] == 1]
    print(f"> xsol: {xsol}")
    print(f"> xsolv: {xsolv}")

    objective_obj = model.obj()
    print(f"> objective_obj: {objective_obj}")

    result = []
    for item in xsol:
        a, b = map(int, item.split("_x_"))
        if a not in result:
            result.append(a)
        if b not in result:
            result.append(b)
    print(f"> result: {result}")
    return result


def multi_target_dijkstra(adj_matrix, source, targets):
    G = nx.DiGraph()
    rows, cols = np.nonzero(adj_matrix)

    for i, j in zip(rows, cols):
        G.add_edge(i, j, weight=adj_matrix[i, j])

    # Run Dijkstra and get full shortest paths
    lengths, paths = nx.single_source_dijkstra(G, source)

    # Get closest target
    closest_target = min(targets, key=lambda t: lengths.get(t, float("inf")))

    print("Closest target:", closest_target)
    print("Path:", paths[closest_target])
    print("Cost:", lengths[closest_target])

    return [int(p) for p in paths[closest_target]]


class SequentialPRM:

    def __init__(self):
        self.rsrc = os.environ["RSRC_DIR"]
        self.graphmlpath = os.path.join(
            self.rsrc, "rnd_torus", "saved_prmstar_planner.graphml"
        )
        self.graphmlmatrixpath = os.path.join(
            self.rsrc, "rnd_torus", "prmstar_euclidean_matrix.npy"
        )
        self.collisionpointpath = os.path.join(
            self.rsrc, "rnd_torus", "collisionpoint_exts.npy"
        )

        self.graphml = nx.read_graphml(self.graphmlpath)
        self.colp = np.load(self.collisionpointpath)
        self.coords = self.euclidean_matrix(self.graphmlmatrixpath)

        # this not actually euclidean data, it is adjacency matrix
        # self.graphmlmatrix = nx.to_numpy_array(self.graphml)

    def euclidean_matrix(self, graphmlmatrixpath):
        if os.path.exists(graphmlmatrixpath):
            return np.load(graphmlmatrixpath)
        coords = np.array(
            [
                list(map(float, self.graphml.nodes[n]["coords"].split(",")))
                for n in self.graphml.nodes
            ]
        )
        np.save(graphmlmatrixpath, coords)
        return coords

    def nearest_node(self, query):
        dist = np.linalg.norm(self.coords - query, axis=1)
        min_idx = np.argmin(dist)
        return min_idx, f"n{min_idx}", self.graphml.nodes[f"n{min_idx}"]

    def euclidean_path_cost(self, path):
        # path lenght cost
        diffs = np.diff(path, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def get_xy(self, node):
        coords = self.graphml.nodes[node]["coords"].split(",")
        return float(coords[0]), float(coords[1])

    def query_path(self, start_np, end_np):
        startindx, start, _ = self.nearest_node(start_np)
        endindx, end, _ = self.nearest_node(end_np)

        try:
            path = nx.shortest_path(self.graphml, source=start, target=end)
            path_coords = [self.get_xy(node) for node in path]
            path_coords.insert(0, (start_np[0], start_np[1]))
            path_coords.append((end_np[0], end_np[1]))
            return np.array(path_coords), self.euclidean_path_cost(path_coords)

        except nx.NetworkXNoPath:
            print(f"no path between {start} and {end}")
            return None, np.inf

    def make_task_seq_matter_adj_matrix(self, task_candidates_num):
        # task_candidates_num = np.array([0, 1, 4, 4, 4])
        node_num = task_candidates_num.sum()
        cs = np.cumsum(task_candidates_num)
        numtask = task_candidates_num.shape[0] - 2

        adj_matrix = np.zeros((node_num, node_num))
        for k in range(numtask):
            for i in range(cs[k], cs[k + 1]):
                for j in range(cs[k + 1], cs[k + 2]):
                    adj_matrix[i, j] = 1
        return adj_matrix

    def make_task_seq_matter_adj_matrix_dist(self, adj_matrix, QQ):
        task_graph_dist = np.zeros_like(adj_matrix)
        for i in range(task_graph_dist.shape[0]):
            for j in range(task_graph_dist.shape[0]):
                if adj_matrix[i, j] != 0:
                    path, cost = self.query_path(QQ[i], QQ[j])
                    if path is not None:
                        task_graph_dist[i, j] = cost
        return task_graph_dist

    def task_seq_matter_graph(self):
        qinit = np.array([0.40, 5.95])
        qtask1 = np.array([-3.26, 5.40])
        qtask2 = np.array([-0.03, 1.08])
        qtask3 = np.array([-2.69, -3.74])

        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

        Qtask1 = Utils.find_alt_config(qtask1.reshape(2, 1), limt2).T
        Qtask2 = Utils.find_alt_config(qtask2.reshape(2, 1), limt2).T
        Qtask3 = Utils.find_alt_config(qtask3.reshape(2, 1), limt2).T
        QQ = np.vstack((qinit, Qtask1, Qtask2, Qtask3))

        task1numcand = Qtask1.shape[0]
        task2numcand = Qtask2.shape[0]
        task3numcand = Qtask3.shape[0]
        task_candidates_num = np.array(
            [0, 1, task1numcand, task2numcand, task3numcand]
        )
        task_graph = self.make_task_seq_matter_adj_matrix(task_candidates_num)
        print(f"task_graph:")
        print(task_graph)

        task_graph_dist = self.make_task_seq_matter_adj_matrix_dist(task_graph, QQ)
        print(f"task_graph_dist:")
        print(task_graph_dist)

        self.plot_graph(None, QQ=QQ)

        startnodeid = 0
        endnodeid = 9
        endnodeset = [9, 10, 11, 12]

        pathid = solve_graph_mip(task_graph_dist, startnodeid, endnodeid)

        pathmulti = multi_target_dijkstra(
            task_graph_dist,
            startnodeid,
            endnodeset,
        )

        path_complete = []
        for i in range(len(pathmulti) - 1):
            path1, _ = self.query_path(QQ[pathmulti[i]], QQ[pathmulti[i + 1]])
            if path1 is not None:
                path_complete.append(path1)

        path_complete = np.vstack(path_complete)
        print(path_complete)

        self.plot_graph(path_complete, QQ)

    def make_task_seq_not_matter_adj_matrix(self, task_candidates_num):
        pass

    def make_task_seq_not_matter_adj_matrix_dist(self, adj_matrix, QQ):
        pass

    def plot_graph(self, path, QQ=None):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-2 * np.pi, 2 * np.pi)

        # tree
        if False:
            for u, v in self.graphml.edges:
                u = self.graphml.nodes[u]["coords"].rsplit(",")
                v = self.graphml.nodes[v]["coords"].rsplit(",")
                ax.plot(
                    [float(u[0]), float(v[0])],
                    [float(u[1]), float(v[1])],
                    color=PlotterConfig.treeColor,
                    linewidth=PlotterConfig.globalLinewidth,
                    marker=PlotterConfig.treeMarker,
                    markerfacecolor=PlotterConfig.treeFaceColor,
                    markersize=PlotterConfig.treeMarkersize,
                )
        if QQ is not None:
            ax.set_title("sequence is blue->red->yellow->green")
            for i in range(QQ.shape[0]):
                if i == 0:
                    color = PlotterConfig.stateStartColor
                if i in [1, 2, 3, 4]:
                    color = "red"
                if i in [5, 6, 7, 8]:
                    color = "yellow"
                if i in [9, 10, 11, 12]:
                    color = "green"

                ax.plot(
                    QQ[i, 0],
                    QQ[i, 1],
                    color=color,
                    linewidth=PlotterConfig.globalLinewidth,
                    marker=PlotterConfig.stateMarker,
                    markerfacecolor=color,
                    markersize=PlotterConfig.stateMarkersize,
                )

        # collision points
        ax.plot(
            self.colp[:, 0],
            self.colp[:, 1],
            color="darkcyan",
            linewidth=0,
            marker="o",
            markerfacecolor="darkcyan",
            markersize=1.5,
        )

        if path is not None:
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=PlotterConfig.pathColor,
                linewidth=PlotterConfig.globalLinewidth,
                marker=PlotterConfig.pathMarker,
                markerfacecolor=PlotterConfig.pathFaceColor,
                markersize=PlotterConfig.pathMarkersize,
            )

        plt.show()


if __name__ == "__main__":
    prm = SequentialPRM()

    # start = np.array([0.0, 0.0])
    # end = np.array([-4, -6])
    # path, cost = prm.query_path(start, end)
    # print("path", path)
    # print("cost", cost)

    # prm.plot_graph(path, QQ=None)

    # prm.query_task()

    prm.task_seq_matter_graph()
