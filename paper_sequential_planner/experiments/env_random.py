import os
import numpy as np
import matplotlib.pyplot as plt
from paper_sequential_planner.scripts.rtsp_solver import RTSP, GLKHHelper
from paper_sequential_planner.scripts.rtsp_lazyprm import (
    separate_sample,
    build_graph,
    estimate_shortest_path,
)


def sample_reachable_wspace(num_points):
    # x,y,z,qx,qy,qz,qw
    H = np.random.uniform(-3, 3, size=(num_points, 7))
    return H


def wspace_ik(ntasks, num_sols, dof):
    # Qaik = np.full((ntasks, num_sols, dof), np.nan)  # (ntasks, num_sols, dof)
    Qaik = np.random.uniform(-np.pi, np.pi, size=(ntasks, num_sols, dof))
    return Qaik


def wspace_ik_validity(Qaik):
    state = np.array([1, -1, -2, -3])
    Qaik_valid = np.random.choice(
        state,
        p=[0.8, 0.1, 0.05, 0.05],
        size=(Qaik.shape[0], Qaik.shape[1], 1),
    )  # (ntasks, num_sols, 1)
    return Qaik_valid


# ------------------- Problem ------------------------------------
def small_size_problem():
    ntasks = 30
    num_sols = 2
    dof = 2
    H = sample_reachable_wspace(ntasks)
    Qaik = wspace_ik(ntasks, num_sols, dof)
    Qaik_valid = wspace_ik_validity(Qaik)
    print(f"==>> H: \n{H}")
    print(f"==>> Qaik.shape: \n{Qaik.shape}")
    print(f"==>> Qaik_valid.shape: \n{Qaik_valid.shape}")
    (
        task_reachable,
        num_qreachable,
        Q_reachable,
        cluster_ttc,
        cluster_ctt,
        taskspace_adjm,
        cspace_adjm,
    ) = RTSP.preprocess(H, Qaik, Qaik_valid)
    print(f"==>> task_reachable: \n{task_reachable}")
    print(f"==>> num_qreachable: \n{num_qreachable}")
    print(f"==>> Q_reachable: \n{Q_reachable}")
    print(f"==>> cluster_ttc: \n{cluster_ttc}")
    print(f"==>> cluster_ctt: \n{cluster_ctt}")
    print(f"==>> taskspace_adjm: \n{taskspace_adjm}")
    print(f"==>> cspace_adjm: \n{cspace_adjm}")

    num_unique_edges = RTSP.num_edges_unique(num_qreachable)
    print(f"==>> num_unique_edges: \n{num_unique_edges}")
    num_supercluster_edges = RTSP.num_supercluster_edges(ntasks)
    print(f"==>> num_supercluster_edges: \n{num_supercluster_edges}")


def mid_size_problem():
    ntasks = 100
    num_sols = 8
    dof = 6
    H = sample_reachable_wspace(ntasks)
    Qaik = wspace_ik(ntasks, num_sols, dof)
    Qaik_valid = wspace_ik_validity(Qaik)
    print(f"==>> H: \n{H}")
    print(f"==>> Qaik.shape: \n{Qaik.shape}")
    print(f"==>> Qaik_valid.shape: \n{Qaik_valid.shape}")
    (
        task_reachable,
        num_qreachable,
        Q_reachable,
        cluster_ttc,
        cluster_ctt,
        taskspace_adjm,
        cspace_adjm,
    ) = RTSP.preprocess(H, Qaik, Qaik_valid)
    print(f"==>> task_reachable: \n{task_reachable}")
    print(f"==>> num_qreachable: \n{num_qreachable}")
    print(f"==>> Q_reachable: \n{Q_reachable}")
    print(f"==>> cluster_ttc: \n{cluster_ttc}")
    print(f"==>> cluster_ctt: \n{cluster_ctt}")
    print(f"==>> taskspace_adjm: \n{taskspace_adjm}")
    print(f"==>> cspace_adjm: \n{cspace_adjm}")

    num_unique_edges = RTSP.num_edges_unique(num_qreachable)
    print(f"==>> num_unique_edges: \n{num_unique_edges}")
    num_supercluster_edges = RTSP.num_supercluster_edges(ntasks)
    print(f"==>> num_supercluster_edges: \n{num_supercluster_edges}")


def large_size_problem():
    # num sols also include torus configuration 8 -> 64
    ntasks = 100
    num_sols = 64
    dof = 6
    H = sample_reachable_wspace(ntasks)
    Qaik = wspace_ik(ntasks, num_sols, dof)
    Qaik_valid = wspace_ik_validity(Qaik)
    print(f"==>> H: \n{H}")
    print(f"==>> Qaik.shape: \n{Qaik.shape}")
    print(f"==>> Qaik_valid.shape: \n{Qaik_valid.shape}")
    (
        task_reachable,
        num_qreachable,
        Q_reachable,
        cluster_ttc,
        cluster_ctt,
        taskspace_adjm,
        cspace_adjm,
    ) = RTSP.preprocess(H, Qaik, Qaik_valid)
    print(f"==>> task_reachable: \n{task_reachable}")
    print(f"==>> num_qreachable: \n{num_qreachable}")
    print(f"==>> Q_reachable: \n{Q_reachable}")
    print(f"==>> cluster_ttc: \n{cluster_ttc}")
    print(f"==>> cluster_ctt: \n{cluster_ctt}")
    print(f"==>> taskspace_adjm: \n{taskspace_adjm}")
    print(f"==>> cspace_adjm: \n{cspace_adjm}")

    num_unique_edges = RTSP.num_edges_unique(num_qreachable)
    print(f"==>> num_unique_edges: \n{num_unique_edges}")
    num_supercluster_edges = RTSP.num_supercluster_edges(ntasks)
    print(f"==>> num_supercluster_edges: \n{num_supercluster_edges}")

def xlarge_size_problem():
    # this is to benchmark againt the other paper
    ntasks = 2000
    num_sols = 8
    dof = 6
    H = sample_reachable_wspace(ntasks)
    Qaik = wspace_ik(ntasks, num_sols, dof)
    Qaik_valid = wspace_ik_validity(Qaik)
    print(f"==>> H: \n{H}")
    print(f"==>> Qaik.shape: \n{Qaik.shape}")
    print(f"==>> Qaik_valid.shape: \n{Qaik_valid.shape}")
    (
        task_reachable,
        num_qreachable,
        Q_reachable,
        cluster_ttc,
        cluster_ctt,
        taskspace_adjm,
        cspace_adjm,
    ) = RTSP.preprocess(H, Qaik, Qaik_valid)
    print(f"==>> task_reachable: \n{task_reachable}")
    print(f"==>> num_qreachable: \n{num_qreachable}")
    print(f"==>> Q_reachable: \n{Q_reachable}")
    print(f"==>> cluster_ttc: \n{cluster_ttc}")
    print(f"==>> cluster_ctt: \n{cluster_ctt}")
    print(f"==>> taskspace_adjm: \n{taskspace_adjm}")
    print(f"==>> cspace_adjm: \n{cspace_adjm}")

    num_unique_edges = RTSP.num_edges_unique(num_qreachable)
    print(f"==>> num_unique_edges: \n{num_unique_edges}")
    num_supercluster_edges = RTSP.num_supercluster_edges(ntasks)
    print(f"==>> num_supercluster_edges: \n{num_supercluster_edges}")

if __name__ == "__main__":
    # small_size_problem()
    # mid_size_problem()
    # large_size_problem()
    xlarge_size_problem()
