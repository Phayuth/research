# RoboTSP + altconfig + better distance metric estimation
from robotsp_solver import RoboTSPSolver
import numpy as np
import time
import util
from plot_ur5e_bullet import UR5eBullet, Constants
import pybullet as p
from exp2 import taskspace_pose, solve_fk_, solve_ik_, solve_ik_altconfig_
import robotsp_solver as rs


def distance_test():
    # default
    rtspsolver = RoboTSPSolver(
        tspace_dist_matrix_func=rs.tspace_distance_matrix_position_euclidean,
        cspace_dist_func=rs.cspace_euclidean_distance,
        tspace_tsp_solver_func=rs.tspace_tsp_solver,
        tspace_tsp_solver_method="heuristic_local_search",
    )

    # case 1: position euclidean orientation geodesic
    rtspsolver = RoboTSPSolver(
        tspace_dist_matrix_func=rs.tspace_distance_matrix_position_euclidean_orient_geodesic,
        cspace_dist_func=rs.cspace_euclidean_distance,
        tspace_tsp_solver_func=rs.tspace_tsp_solver,
        tspace_tsp_solver_method="heuristic_local_search",
    )

    # case 2: ...more...

    Htasks = taskspace_pose()
    # qinit = np.array([3.4, -1.22, 1.25, 0.0, 1.81, 0.0]) # altconfig better
    # qinit = np.array([0.0, -1.22, 1.25, 0.0, 1.81, 0.0]) # altconfig better
    qinit = np.array([7.1, -1.22, 1.25, 0.0, 1.81, 0.0])  # normal config better
    Hinit = solve_fk_(qinit)
    numsolslist, Qlist = solve_ik_(Htasks)
    rtspsolver.solve(Htasks, Hinit, qinit, numsolslist, Qlist)
    rtspsolver.print_log()


if __name__ == "__main__":
    distance_test()
