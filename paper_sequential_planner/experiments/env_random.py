import os
import numpy as np
import matplotlib.pyplot as plt
from paper_sequential_planner.scripts.rtsp_solver import RTSP, GLKHHelper
from paper_sequential_planner.scripts.rtsp_lazyprm import (
    separate_sample,
    build_graph,
    estimate_shortest_path,
)


def example_usage():
    # problem setup
    dof = 2
    points_per_cluster = [3, 3, 3, 2]
    num_node = sum(points_per_cluster)
    q0 = np.zeros((dof,))
    config = np.random.uniform(-np.pi, np.pi, size=(num_node, dof))
    # H 4x4 matrix flatten row-major
    H = np.random.uniform(-3, 3, size=(len(points_per_cluster), 16))

    # compute GTSP data
    cluster = RTSP.build_cluster_task_to_cspace(points_per_cluster)
    adjm = RTSP.build_cspace_adjm(cluster, num_node)
    num_unique_edges = RTSP.find_numedges_unique(points_per_cluster)
    print("cluster:", cluster)
    print("adjm:\n", adjm)
    print("num_unique_edges:", num_unique_edges)

    edge_cost_distance = RTSP.edgecost_eucl_distance(config)
    print(f"==>> edge_cost_distance: \n{edge_cost_distance}")
    GLKHHelper.write_glkh_fullmatrix_file(
        os.path.join(GLKHHelper.problemdir, "problem2_fullmatrix.gtsp"),
        edge_cost_distance,
        cluster,
    )

    # solve GTSP using GLKH
    if os.path.exists(
        os.path.join(GLKHHelper.problemdir, "problem2_fullmatrix.tour")
    ):
        tourmatix = GLKHHelper.read_tour_file(
            os.path.join(GLKHHelper.problemdir, "problem2_fullmatrix.tour")
        )
        print(f"==>> tourmatix: \n{tourmatix}")

        fig, ax = plt.subplots()
        for key, c in cluster.items():
            c = np.array(c)
            ax.scatter(config[c, 0], config[c, 1], label=f"cluster {key}")
        for i in range(len(tourmatix) - 1):
            start_idx = tourmatix[i]
            end_idx = tourmatix[i + 1]
            ax.plot(
                [config[start_idx, 0], config[end_idx, 0]],
                [config[start_idx, 1], config[end_idx, 1]],
                c="red",
                label="tour" if i == 0 else None,
            )
        ax.set_title("GTSP Tour from GLKH")
        ax.legend()
        plt.show()
    else:
        print("Tour file not found. Please run GLKH solver file.")


if __name__ == "__main__":
    example_usage()
