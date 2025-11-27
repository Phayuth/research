import numpy as np
import matplotlib.pyplot as plt
import os
from gtsp_ultra import GUUtil, GU

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)


class RandomCircleGen:
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
    def cluster_center_in_circle(xc=0, yc=0, r=1, n=1, thetashift=0):
        centers = []
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False) + thetashift
        for theta in thetas:
            center_x = xc + r * np.cos(theta)
            center_y = yc + r * np.sin(theta)
            centers.append((center_x, center_y))
        return centers

    @staticmethod
    def generate(
        num_clusters,
        each_cluster,
        center_radius=1,
        cluster_radius=0.5,
        thetashift=0,
    ):
        _cluster_centers = RandomCircleGen.cluster_center_in_circle(
            xc=0,
            yc=0,
            r=center_radius,
            n=num_clusters,
            thetashift=thetashift,
        )
        config = []
        for center in _cluster_centers:
            points = RandomCircleGen.random_point_in_circle(
                x=center[0],
                y=center[1],
                r=cluster_radius,
                n=each_cluster,
            )
            config.append(points)
        config = np.vstack(config)
        points_per_cluster = [each_cluster] * num_clusters
        return config, points_per_cluster, _cluster_centers

    @staticmethod
    def plot(all_points, tour=None, cluster_centers=[], cluster_radius=0.5):
        all_points = np.vstack(all_points)

        plt.figure()
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


if __name__ == "__main__":
    # problem setup
    num_clusters = 4
    each_cluster = 8
    cluster_radius = 0.2
    center_radius = 2.5
    config, points_per_cluster, cluster_centers = RandomCircleGen.generate(
        num_clusters=num_clusters,
        each_cluster=each_cluster,
        center_radius=center_radius,
        cluster_radius=cluster_radius,
        thetashift=np.pi / 4,
    )

    # compute GTSP data
    cluster = GU.build_cluster(points_per_cluster)
    cost_adj_matrix = GU.edgecost_distance(config)

    GUUtil.write_glkh_fullmatrix_file(
        os.path.join(
            GUUtil.problemdir,
            f"{num_clusters}random{num_clusters*each_cluster}.gtsp",
        ),
        cost_adj_matrix,
        cluster,
    )
    # # solve GTSP using GLKH
    # tourmatrix = GUUtil.read_tour_file(
    #     os.path.join(GUUtil.problemdir, "random_gtsp_fullmatrix.tour")
    # )

    # plot result
    RandomCircleGen.plot(
        config,
        tour=None,
        cluster_centers=cluster_centers,
        cluster_radius=cluster_radius,
    )
