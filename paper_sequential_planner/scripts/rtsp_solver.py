import os
import numpy as np
from itertools import product

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class RTSP:

    def __init__(self):
        pass

    @staticmethod
    def build_cluster_task_to_cspace(points_per_cluster):
        """
        Building cluster mapping from task to cspace.
        Easy to access the cspace points given a task.
        """
        cluster = {
            i: list(
                range(
                    sum(points_per_cluster[:i]),
                    sum(points_per_cluster[: i + 1]),
                )
            )
            for i in range(len(points_per_cluster))
        }
        return cluster

    @staticmethod
    def build_cluster_cspace_to_task(points_per_cluster):
        """
        Building cluster mapping from cspace to task.
        Easy to access the task given a cspace point.
        """
        cluster = {}
        current_idx = 0
        for task_idx, num_sols in enumerate(points_per_cluster):
            for sol_idx in range(int(num_sols)):
                cluster[current_idx] = task_idx
                current_idx += 1
        return cluster

    @staticmethod
    def build_taskspace_adjm(n_tasks):
        """
        Building task space adjacency matrix.
         -1: same task,
          0: different task

        """
        taskspace_adjm = np.full((n_tasks, n_tasks), -1, dtype=int)
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                if i == j:
                    continue
                else:
                    taskspace_adjm[i, j] = 0
                    taskspace_adjm[j, i] = 0
        return taskspace_adjm

    @staticmethod
    def build_cspace_adjm(cluster, num_sols):
        cspace_adjm = np.full((num_sols, num_sols), 0.0)
        for c in cluster.values():
            for i in c:
                for j in c:
                    cspace_adjm[i, j] = -1.0
        return cspace_adjm

    @staticmethod
    def check_connection(cluster, cspace_adjm, task1, task2):
        c_task1 = cluster[task1]
        c_task2 = cluster[task2]
        l = list(product(c_task1, c_task2))
        cc = []
        for i, j in l:
            cc.append(cspace_adjm[i, j].item())
        return cc

    @staticmethod
    def find_numedges_unique(points_per_cluster):
        totalnode = sum(points_per_cluster)
        self_connections = sum([n * (n - 1) / 2 for n in points_per_cluster])
        unique_edges = (totalnode * (totalnode - 1) / 2) - self_connections
        return unique_edges

    @staticmethod
    def edgecost_distance(config):
        num_node = config.shape[0]
        cost = np.full((num_node, num_node), 0.0)
        for i in range(num_node):
            for j in range(num_node):
                if i != j:
                    diff = config[i] - config[j]
                    cost[i, j] = np.linalg.norm(diff)
        return cost

    @staticmethod
    def edgecost_colfreepath_est(config):
        pass


class GLKHHelper:
    rsrc_dir = os.environ["RSRC_DIR"]
    problemdir = os.path.join(rsrc_dir, "GLKH-1.1", "PROBLEMS")

    @staticmethod
    def write_glkh_coord_display_file(filename, all_points, clusters):
        num_points = all_points.shape[0]
        num_clusters = len(clusters.keys())
        with open(filename, "w") as f:
            f.write(f"NAME: random_gtsp\n")
            f.write(f"TYPE: GTSP\n")
            f.write(f"COMMENT: generated random GTSP instance\n")
            f.write(f"DIMENSION: {num_points}\n")
            f.write(f"GTSP_SETS: {num_clusters}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FUNCTION\n")
            f.write(f"DISPLAY_DATA_TYPE: COORD_DISPLAY\n")
            f.write(f"NODE_COORD_SECTION\n")

            # --- write all points ---
            idx = 1
            for q in all_points:
                q = np.asarray(q).ravel()
                coords = [float(x) for x in q]
                coords_str = " ".join(f"{c:.6f}" for c in coords)
                f.write(f"{idx:4d} {coords_str}\n")
                idx += 1

            # --- write GTSP sets ---
            f.write("GTSP_SET_SECTION\n")
            for key in clusters.keys():
                c = clusters[key]
                k = key
                nodes_str = " ".join(str(n + 1) for n in c)
                f.write(f"{k + 1} {nodes_str} -1\n")

            f.write("EOF\n")

    @staticmethod
    def write_glkh_fullmatrix_file(filename, matrix, clusters):
        matrix = matrix * 1000  # scale to convert float to int
        matrix = matrix.astype(int)
        num_points = matrix.shape[0]
        num_clusters = len(clusters.keys())
        with open(filename, "w") as f:
            f.write(f"NAME: random_gtsp_fullmatrix\n")
            f.write(f"TYPE: GTSP\n")
            f.write(f"COMMENT: generated GTSP/AGTSP instance with full matrix\n")
            f.write(f"DIMENSION: {num_points}\n")
            f.write(f"GTSP_SETS: {num_clusters}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write(f"EDGE_WEIGHT_SECTION\n")

            # --- write all points ---
            for i in range(num_points):
                row = " ".join(f"{matrix[i, j]}" for j in range(num_points))
                f.write(f"{row}\n")

            # --- write GTSP sets ---
            f.write("GTSP_SET_SECTION\n")
            for key in clusters.keys():
                c = clusters[key]
                k = key
                nodes_str = " ".join(str(n + 1) for n in c)
                f.write(f"{k + 1} {nodes_str} -1\n")

            f.write("EOF\n")

    @staticmethod
    def read_tour_file(filename):
        tour = []
        with open(filename, "r") as f:
            lines = f.readlines()
            reading_tour = False
            for line in lines:
                line = line.strip()
                if line == "TOUR_SECTION":
                    reading_tour = True
                    continue
                if line == "-1" or line == "EOF":
                    break
                if reading_tour:
                    idx = int(line) - 1  # convert to 0-based index
                    tour.append(idx)
        tour.append(tour[0])  # make it a round trip
        return np.array(tour)


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

    edge_cost_distance = RTSP.edgecost_distance(config)
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

        import matplotlib.pyplot as plt

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
