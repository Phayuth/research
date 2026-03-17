import os
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


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
    from paper_sequential_planner.scripts.rtsp_solver import RTSP, GLKHHelper

    # problem setup
    num_clusters = 30
    each_cluster = 8
    cluster_radius = 0.2
    center_radius = 2.5
    Q_reachable, num_qreachable, cluster_centers = RandomCircleGen.generate(
        num_clusters=num_clusters,
        each_cluster=each_cluster,
        center_radius=center_radius,
        cluster_radius=cluster_radius,
        thetashift=np.pi / 4,
    )
    print(f"==>> Q_reachable: \n{Q_reachable}")
    print(f"==>> num_qreachable: \n{num_qreachable}")
    print(f"==>> cluster_centers: \n{cluster_centers}")

    # compute GTSP data
    cluster_ttc = RTSP.build_cluster_task_to_cspace(num_qreachable)
    num_sols = sum(num_qreachable)
    cspace_adjm = RTSP.build_cspace_adjm(cluster_ttc, num_sols)
    print(f"==>> num_sols: \n{num_sols}")
    print(f"==>> cluster_ttc: \n{cluster_ttc}")
    print(f"==>> cspace_adjm: \n{cspace_adjm}")

    cspace_adjm_euc_min = RTSP.edgecost_eucl_distance(Q_reachable)
    print(f"==>> cspace_adjm_euc_min: \n{cspace_adjm_euc_min}")

    GLKHHelper.write_glkh_fullmatrix_file(
        os.path.join(GLKHHelper.problemdir, "problem_randomcircle.gtsp"),
        cspace_adjm_euc_min,
        cluster_ttc,
    )

    tour = None
    if os.path.exists(os.path.join(GLKHHelper.problemdir, "problem_randomcircle.tour")):
        tour = GLKHHelper.read_tour_file(
            os.path.join(GLKHHelper.problemdir, "problem_randomcircle.tour")
        )
        print(f"==>> tour: \n{tour}")

    RandomCircleGen.plot(
        all_points=Q_reachable,
        tour=tour,
        cluster_centers=cluster_centers,
        cluster_radius=cluster_radius,
    )
