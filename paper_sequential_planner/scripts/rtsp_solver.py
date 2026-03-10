import os
import numpy as np
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances

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
        cluster_ttc = {
            i: list(
                range(
                    sum(points_per_cluster[:i]),
                    sum(points_per_cluster[: i + 1]),
                )
            )
            for i in range(len(points_per_cluster))
        }
        return cluster_ttc

    @staticmethod
    def build_cluster_cspace_to_task(points_per_cluster):
        """
        Building cluster mapping from cspace to task.
        Easy to access the task given a cspace point.
        """
        cluster_ctt = {}
        current_idx = 0
        for task_idx, num_sols in enumerate(points_per_cluster):
            for sol_idx in range(int(num_sols)):
                cluster_ctt[current_idx] = task_idx
                current_idx += 1
        return cluster_ctt

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
    def update_taskspace_adjm(taskspace_adjm, cspace_adjm, cspace_to_taskspace):
        """Update taskspace adjm by counting cspace edges between task pairs"""
        for i in range(cspace_adjm.shape[0]):
            for j in range(i + 1, cspace_adjm.shape[1]):
                if cspace_adjm[i, j] != -1.0:  # Valid connection in cspace
                    task_i = cspace_to_taskspace[i]
                    task_j = cspace_to_taskspace[j]
                    if task_i != task_j:  # Only count inter-task connections
                        taskspace_adjm[task_i, task_j] += 1
                        taskspace_adjm[task_j, task_i] += 1
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
    def edgecost_eucl_distance(config):
        cost = euclidean_distances(config, config)
        np.fill_diagonal(cost, -1.0)
        return cost

    @staticmethod
    def edgecost_colfree_distance(
        cspace_adjm,
        config,
        estimator,
        estimator_params,
    ):
        """Here estimation can fail due to sampling nature and graph is disconnected"""
        store_path = {}
        store_cost = {}
        for i in range(cspace_adjm.shape[0]):
            for j in range(i + 1, cspace_adjm.shape[1]):
                if cspace_adjm[i, j] != -1.0:
                    q1 = config[i]
                    q2 = config[j]
                    res_ = estimator(q1, q2, **estimator_params)
                    if res_ is not None:
                        pathq, cost = res_
                        cspace_adjm[i, j] = cost
                        cspace_adjm[j, i] = cost
                        store_path[(i, j)] = pathq
                        store_cost[(i, j)] = cost
                        print(f"Estimated cost from {i} to {j}: {cost}")
                    else:
                        cspace_adjm[i, j] = -2.0
                        cspace_adjm[j, i] = -2.0
                        store_path[(i, j)] = np.nan
                        store_cost[(i, j)] = np.inf
                        print(f"No path found from {i} to {j}, fail to est cost.")
        return cspace_adjm, store_path, store_cost

    @staticmethod
    def preprocess(taskH, Qaik, Qaik_valid):
        dof = Qaik.shape[2]
        task_reachablemask = np.any(Qaik_valid == 1, axis=1).flatten()  # (ntasks,)
        q_reachable_perH = np.sum(Qaik_valid == 1, axis=1).flatten()  # (ntasks,)
        num_reachable = np.sum(task_reachablemask)
        task_reachable = taskH[task_reachablemask]
        num_qreachable = q_reachable_perH[q_reachable_perH > 0]
        _Qaik_flat = Qaik.reshape(-1, dof)  # (ntasks * num_solutions, dof)
        _Qaik_valid_flat = Qaik_valid.reshape(-1)  # (ntasks * num_solutions,)
        Q_reachable = _Qaik_flat[_Qaik_valid_flat == 1]
        taskspace_adjm = RTSP.build_taskspace_adjm(num_reachable)
        cluster_ttc = RTSP.build_cluster_task_to_cspace(num_qreachable)
        num_sols = sum(num_qreachable)
        cspace_adjm = RTSP.build_cspace_adjm(cluster_ttc, num_sols)
        return (
            task_reachable,
            num_qreachable,
            Q_reachable,
            cluster_ttc,
            taskspace_adjm,
            cspace_adjm,
        )

    @staticmethod
    def postprocess(tourid, Q_reachable, colfree_planner):
        """
        tourid: the tour, e.g., [0, 2, 1, 0] (return back to start)
        Q_reachable: the reachable configurations
        """
        qtour = Q_reachable[tourid]
        store_path = {}
        store_cost = {}
        for i in range(len(tourid) - 1):
            start_idx = tourid[i]
            end_idx = tourid[i + 1]
            q1 = Q_reachable[start_idx]
            q2 = Q_reachable[end_idx]
            res_ = colfree_planner(q1, q2)
            if res_ is not None:
                qp, cp = res_
                qp = np.array(qp)
                store_path[(start_idx, end_idx)] = qp
                store_cost[(start_idx, end_idx)] = cp
            else:
                store_path[(start_idx, end_idx)] = np.nan
                store_cost[(start_idx, end_idx)] = np.inf
        return qtour, store_path, store_cost


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
