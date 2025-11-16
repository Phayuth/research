# dev code for RTSP paper
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)


# ============== Data Generation ==============
def random_point_in_circle(x, y, r=0.5, n=1):
    points = []
    for _ in range(n):
        theta = np.random.uniform(0, 2 * np.pi)
        radius = r * np.sqrt(np.random.uniform(0, 1))
        point_x = x + radius * np.cos(theta)
        point_y = y + radius * np.sin(theta)
        points.append((point_x, point_y))
    return np.array(points)


def cluster_center_in_circle(xc=0, yc=0, r=1, n=1):
    centers = []
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for theta in thetas:
        center_x = xc + r * np.cos(theta)
        center_y = yc + r * np.sin(theta)
        centers.append((center_x, center_y))
    return centers


def write_glkh_file(filename, clusters):
    num_clusters = len(clusters)
    num_points = sum(len(c) for c in clusters)

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
        point_indices = []
        for cluster in clusters:
            cluster_indices = []
            for x, y in cluster:
                f.write(f"{idx:4d} {x:.6f} {y:.6f}\n")
                cluster_indices.append(idx)
                idx += 1
            point_indices.append(cluster_indices)

        # --- write GTSP sets ---
        f.write("GTSP_SET_SECTION\n")
        for i, cluster_indices in enumerate(point_indices, start=1):
            cluster_str = " ".join(map(str, cluster_indices))
            f.write(f"{i} {cluster_str} -1\n")

        f.write("EOF\n")


def plot():
    cluster_centers = cluster_center_in_circle(n=5)
    cluster_radius = 0.5
    all_points = []
    for i in range(num_clusters):
        center = cluster_centers[i]
        n_points = points_per_cluster[i]
        points = random_point_in_circle(
            center[0], center[1], r=cluster_radius, n=n_points
        )
        all_points.append(points)
    # write_glkh_file("random_gtsp.gtsp", all_points)
    all_points = np.vstack(all_points)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_points[:, 0], all_points[:, 1], c="blue")
    # draw tour
    # tour_id = [2, 18, 5, 15, 9, 2]
    # for i in range(len(tour_id) - 1):
    #     p1 = all_points[tour_id[i] - 1]
    #     p2 = all_points[tour_id[i + 1] - 1]
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "g--")
    for center in cluster_centers:
        circle = plt.Circle(
            center, cluster_radius, color="red", fill=False, linestyle="--"
        )
        plt.gca().add_artist(circle)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Random Points in Clusters")
    plt.grid()
    plt.show()
    print("Generated Points:\n", all_points)


# user inputs
points_per_cluster = [3, 4, 2, 4, 3]
num_node = sum(points_per_cluster)
config = np.random.uniform(-np.pi, np.pi, size=(num_node, 6))
# H 4x4 matrix flatten row-major
H = np.random.uniform(-3, 3, size=(len(points_per_cluster), 16))


# prepare data
idx = list(range(num_node))
cluster = {
    i: list(
        range(
            sum(points_per_cluster[:i]),
            sum(points_per_cluster[: i + 1]),
        )
    )
    for i in range(len(points_per_cluster))
}
print("cluster:", cluster)


def make_adj_matrix():
    adjm = np.full((num_node, num_node), np.inf)
    for c in cluster.values():
        for i in c:
            for j in c:
                adjm[i, j] = -1
    return adjm


def cost_collision(config):
    node_cost = np.zeros((config.shape[0],))
    # dummy collision cost
    for i in range(config.shape[0]):
        if np.random.rand() < 0.3:
            node_cost[i] = np.inf  # collision
        else:
            node_cost[i] = 0.0  # free
    return node_cost


def cost_manipulability(config):
    node_cost = np.zeros((config.shape[0],))
    # dummy manipulability cost
    for i in range(config.shape[0]):
        node_cost[i] = 1.0 - np.abs(config[i, 5]) / np.pi  # prefer elbow-down
    return node_cost


def cost_similarity(config):
    node_cost = np.zeros((config.shape[0],))
    # dummy similarity cost
    for i in range(config.shape[0]):
        node_cost[i] = np.linalg.norm(config[i, :3]) / (
            3 * np.pi
        )  # prefer close to origin
    return node_cost


def cost_total(config):
    return (
        cost_collision(config)
        + cost_manipulability(config)
        + cost_similarity(config)
    )


def edgecost_distance(config):
    num_node = config.shape[0]
    cost = np.full((num_node, num_node), np.inf)
    for i in range(num_node):
        for j in range(num_node):
            if i != j:
                diff = config[i] - config[j]
                cost[i, j] = np.linalg.norm(diff)
    return cost


node_score = cost_total(config)
print("node_score:", node_score)

adjm = make_adj_matrix()

cost = adjm.copy()
