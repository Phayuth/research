import numpy as np
import matplotlib.pyplot as plt
import os
from gtsp_ultra import GUUtil, GU

if __name__ == "__main__":
    # problem setup
    dof = 2
    points_per_cluster = [3, 3, 3]
    num_node = sum(points_per_cluster)
    q0 = np.zeros((dof,))
    config = np.random.uniform(-np.pi, np.pi, size=(num_node, dof))
    # H 4x4 matrix flatten row-major
    H = np.random.uniform(-3, 3, size=(len(points_per_cluster), 16))

    # compute GTSP data
    cluster = GU.build_cluster(points_per_cluster)
    adjm = GU.make_adj_matrix(cluster, num_node)
    num_unique_edges = GU.find_numedges_unique(points_per_cluster)
    print("cluster:", cluster)
    print("adjm:\n", adjm)
    print("num_unique_edges:", num_unique_edges)

    node_cost_collision = GU.nodecost_collision(config)
    node_cost_manipulability = GU.nodecost_manipulability(config)
    node_cost_similarity = GU.nodecost_similarity(config, q0)

    edge_cost_distance = GU.edgecost_distance(config)

    # GUUtil.write_glkh_fullmatrix_file(
    #     os.path.join(GUUtil.problemdir, "problem2_fullmatrix.gtsp"),
    #     edge_cost_distance,
    #     cluster,
    # )

    # # solve GTSP using GLKH
    # tourmatix = GUUtil.read_tour_file(
    #     os.path.join(GUUtil.problemdir, "problem2_fullmatrix.tour")
    # )
