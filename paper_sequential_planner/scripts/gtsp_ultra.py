# dev code for RTSP paper
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)


class FakeDataGenerator:
    @staticmethod
    def random_point_in_circle(x, y, r=0.5, n=1):
        points = []
        for _ in range(n):
            theta = np.random.uniform(0, 2 * np.pi)
            radius = r * np.sqrt(np.random.uniform(0, 1))
            point_x = x + radius * np.cos(theta)
            point_y = y + radius * np.sin(theta)
            points.append((point_x, point_y))
        return np.array(points)

    @staticmethod
    def cluster_center_in_circle(xc=0, yc=0, r=1, n=1):
        centers = []
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for theta in thetas:
            center_x = xc + r * np.cos(theta)
            center_y = yc + r * np.sin(theta)
            centers.append((center_x, center_y))
        return centers

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
        matrix = matrix * 100000  # scale to convert float to int
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

    @staticmethod
    def plot(all_points, tour=None, cluster_centers=[], cluster_radius=0.5):
        all_points = np.vstack(all_points)

        plt.figure(figsize=(8, 8))
        plt.scatter(all_points[:, 0], all_points[:, 1], c="blue")
        if tour is not None:
            for i in range(len(tour) - 1):
                p1 = all_points[tour[i]]
                p2 = all_points[tour[i + 1]]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "g--")
        for center in cluster_centers:
            circle = plt.Circle(
                center, cluster_radius, color="red", fill=False, linestyle="--"
            )
            plt.gca().add_artist(circle)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid()
        plt.show()


# =================== GTSP Robotics Dev Code ===================
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


def make_adj_matrix(cluster, num_node):
    adjm = np.full((num_node, num_node), np.inf)
    for c in cluster.values():
        for i in c:
            for j in c:
                adjm[i, j] = -1
    return adjm


def find_numedges_unique(points_per_cluster):
    totalnode = sum(points_per_cluster)
    self_connections = sum([n * (n - 1) / 2 for n in points_per_cluster])
    unique_edges = (totalnode * (totalnode - 1) / 2) - self_connections
    return unique_edges


def nodecost_collision(config):
    node_cost = np.zeros((config.shape[0],))
    # dummy collision cost
    for i in range(config.shape[0]):
        if np.random.rand() < 0.3:
            node_cost[i] = np.inf  # collision
        else:
            node_cost[i] = 0.0  # free
    return node_cost


def nodecost_sdf(config):
    """compute signed distance function"""
    node_cost = np.zeros((config.shape[0],))
    # dummy sdf cost
    for i in range(config.shape[0]):
        node_cost[i] = np.random.uniform(0.0, 1.0)
    return node_cost


def nodecost_manipulability(config):
    node_cost = np.zeros((config.shape[0],))
    # dummy manipulability cost
    for i in range(config.shape[0]):
        node_cost[i] = np.random.uniform(0.0, 1.0)
    return node_cost


def nodecost_similarity(config, q0):
    """compute similarity cost to a reference pose"""
    node_cost = np.zeros((config.shape[0],))
    # dummy similarity cost
    for i in range(config.shape[0]):
        node_cost[i] = np.random.uniform(0.0, 1.0)
    return node_cost


def edgecost_distance(config):
    num_node = config.shape[0]
    cost = np.full((num_node, num_node), 0.0)
    for i in range(num_node):
        for j in range(num_node):
            if i != j:
                diff = config[i] - config[j]
                cost[i, j] = np.linalg.norm(diff)
    return cost


def edgecost_colfreepath_est(config):
    pass


def solve_tour():
    pass


if __name__ == "__main__":
    # # user inputs
    # dof = 2
    # print("dof:", dof)
    # points_per_cluster = [3, 3, 2]
    # print("points_per_cluster:", points_per_cluster)
    # num_node = sum(points_per_cluster)
    # print("num_node:", num_node)
    # q0 = np.zeros((dof,))
    # print("q0:", q0)
    # config = np.random.uniform(-np.pi, np.pi, size=(num_node, dof))
    # print("config:\n", config)
    # # H 4x4 matrix flatten row-major
    # H = np.random.uniform(-3, 3, size=(len(points_per_cluster), 16))
    # print("H:\n", H)

    # cluster = build_cluster(points_per_cluster)
    # print("cluster:\n", cluster)
    # adjm = make_adj_matrix(cluster)
    # print("adjacency matrix:\n", adjm)
    # num_unique_edges = find_numedges_unique(points_per_cluster)
    # print("num unique edges:", num_unique_edges)

    # node_cost_collision = nodecost_collision(config)
    # print("node cost (collision):\n", node_cost_collision)
    # node_cost_manipulability = nodecost_manipulability(config)
    # print("node cost (manipulability):\n", node_cost_manipulability)
    # node_cost_similarity = nodecost_similarity(config, q0)
    # print("node cost (similarity):\n", node_cost_similarity)

    # edge_cost_distance = edgecost_distance(config)
    # print("edge cost (distance):\n", edge_cost_distance)

    # parameters
    num_clusters = 24
    points_per_cluster = 8
    cluster_radius = 0.5
    cluster_centers = FakeDataGenerator.cluster_center_in_circle(
        xc=0, yc=0, r=3, n=num_clusters
    )
    all_points = []
    for center in cluster_centers:
        points = FakeDataGenerator.random_point_in_circle(
            x=center[0], y=center[1], r=cluster_radius, n=points_per_cluster
        )
        all_points.append(points)

    all_points = np.vstack(all_points)
    points_per_cluster = [points_per_cluster] * num_clusters
    cluster = build_cluster(points_per_cluster)
    num_node = sum(points_per_cluster)
    print("all_points:\n", all_points)
    print("points_per_cluster:", points_per_cluster)
    print("cluster:\n", cluster)

    cost_adj_matrix = edgecost_distance(all_points)
    print("cost_adj_matrix:\n", cost_adj_matrix)

    rsrc_dir = os.environ["RSRC_DIR"]
    problemdir = os.path.join(rsrc_dir, "GLKH-1.1", "PROBLEMS")

    FakeDataGenerator.write_glkh_coord_display_file(
        os.path.join(problemdir, "random_gtsp.gtsp"), all_points, cluster
    )
    FakeDataGenerator.write_glkh_fullmatrix_file(
        os.path.join(problemdir, "random_gtsp_fullmatrix.gtsp"),
        cost_adj_matrix,
        cluster,
    )
    tourmatrix = FakeDataGenerator.read_tour_file(
        os.path.join(problemdir, "random_gtsp_fullmatrix.tour")
    )
    FakeDataGenerator.plot(
        all_points,
        tour=tourmatrix,
        cluster_centers=cluster_centers,
        cluster_radius=cluster_radius,
    )
