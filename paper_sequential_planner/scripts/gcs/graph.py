import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(9)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def visualize_bidirectional_graph(matrix):
    G = nx.Graph()

    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):  # Avoid duplication of edges in bidirectional graph
            if matrix[i][j] != 0:
                G.add_edge(i, j, weight=matrix[i][j])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    edge_labels = nx.get_edge_attributes(G, "weight")
    pos = nx.spring_layout(G)  # euclidean positions for nodes
    nx.draw_networkx(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    ax.set_aspect("equal")
    ax.margins(0.20)
    ax.grid(False)
    plt.show()


def solve_graph_mip(graph):
    from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, NonNegativeReals, Integers, minimize, Binary, ConstraintList
    import re

    # this looks exactly like network flow problem solved using LP
    startnode = 0
    endnode = 5
    model = ConcreteModel()

    A = []
    w = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            # add any graph edge that is not 0 and tail is not startnode and head is not endnode
            if graph[i][j] != 0 and j != startnode and i != endnode:
                A.append(f"x{i}{j}")
                w.append(graph[i][j])

    model.x = Var(A, within=Binary)
    model.obj = Objective(expr=sum(w[i] * model.x[e] for i, e in enumerate(A)), sense=minimize)
    model.constraints = ConstraintList()

    print(A)
    print(w)

    startpattern = rf"^x{startnode}.*"
    endpattern = rf"^x.*{endnode}$"

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
        sp = rf"^x{i}.*"
        ep = rf"^x.*{i}$"
        sc = [s for s in A if re.match(sp, s)]
        print(f"> sc: {sc}")
        ec = [s for s in A if re.match(ep, s)]
        print(f"> ec: {ec}")
        if len(sc) == 0 or len(ec) == 0:
            continue
        model.constraints.add(expr=sum(model.x[xi] for xi in sc) == sum(model.x[xi] for xi in ec))
        print("add constraint")

    opt = SolverFactory("glpk")
    result_obj = opt.solve(model, tee=True)

    model.pprint()

    opt_solution = [model.x[item].value for item in A]
    print(f"> opt_solution: {opt_solution}")

    xsol = [e for i, e in enumerate(A) if opt_solution[i] == 1]
    print(f"> xsol: {xsol}")

    objective_obj = model.obj()
    print(f"> objective_obj: {objective_obj}")


if __name__ == "__main__":
    graph = np.array(
        [
            [0, 35, 30, 20, 0, 0, 0],
            [35, 0, 8, 0, 12, 0, 0],
            [30, 8, 0, 9, 10, 20, 0],
            [20, 0, 9, 0, 0, 0, 15],
            [0, 12, 10, 0, 0, 5, 20],
            [0, 0, 20, 0, 5, 0, 5],
            [0, 0, 0, 15, 20, 5, 0],
        ]
    )
    solve_graph_mip(graph)
    visualize_bidirectional_graph(graph)
