# dev code for RTSP paper
import numpy as np
import os

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)


class GUUtil:
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


class GU:
    @staticmethod
    def build_cluster(points_per_cluster):
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
    def make_adj_matrix(cluster, num_node):
        adjm = np.full((num_node, num_node), np.inf)
        for c in cluster.values():
            for i in c:
                for j in c:
                    adjm[i, j] = -1
        return adjm

    @staticmethod
    def find_numedges_unique(points_per_cluster):
        totalnode = sum(points_per_cluster)
        self_connections = sum([n * (n - 1) / 2 for n in points_per_cluster])
        unique_edges = (totalnode * (totalnode - 1) / 2) - self_connections
        return unique_edges

    @staticmethod
    def nodecost_collision(config):
        node_cost = np.zeros((config.shape[0],))
        # dummy collision cost
        for i in range(config.shape[0]):
            if np.random.rand() < 0.3:
                node_cost[i] = np.inf  # collision
            else:
                node_cost[i] = 0.0  # free
        return node_cost

    @staticmethod
    def nodecost_sdf(config):
        """compute signed distance function"""
        node_cost = np.zeros((config.shape[0],))
        # dummy sdf cost
        for i in range(config.shape[0]):
            node_cost[i] = np.random.uniform(0.0, 1.0)
        return node_cost

    @staticmethod
    def nodecost_manipulability(config):
        node_cost = np.zeros((config.shape[0],))
        # dummy manipulability cost
        for i in range(config.shape[0]):
            node_cost[i] = np.random.uniform(0.0, 1.0)
        return node_cost

    @staticmethod
    def nodecost_similarity(config, q0):
        """compute similarity cost to a reference pose"""
        node_cost = np.zeros((config.shape[0],))
        # dummy similarity cost
        for i in range(config.shape[0]):
            node_cost[i] = np.random.uniform(0.0, 1.0)
        return node_cost

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