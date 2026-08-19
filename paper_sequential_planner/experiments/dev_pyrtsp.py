import os
import numpy as np
import time
import tqdm
import torch
from paper_sequential_planner.scripts.geometric_torus import find_alt_config2
from paper_sequential_planner.scripts.geometric_poses import (
    H_to_X,
    Hlist_to_Xlist,
    Xlist_to_Hlist,
    Naive_task_space_correlation,
    KRNN_task_space_correlation,
    Cluster_task_space_correlation,
    taskspace_tsp_position_distance_order,
    taskspace_brute_permutation_order,
    nn_dict_to_adjmat,
)
from paper_sequential_planner.scripts.geometric_config import (
    traj_complete_cost,
    traj_tour_from_lininterp,
    traj_tour_from_lininterp_qdot,
)
from paper_sequential_planner.scripts.geometric_rtsp import (
    Qfilter_R,
    Qfilter_similarity,
    Qfilter_nn2c,
    Qfilter_Knn2c,
    Qfilter_Dnn2c,
    Qfilter_ClusterTSP,
    Eest_colfree,
    Eest_weighted_euclidean,
    Eest_weighted_max_joint_diff,
    Qtour_RoboTSP_layer_search,
    Qtour_RoboTSP_layer_search_topK,
)
from paper_sequential_planner.experiments.env_ur5e_ import (
    RobotUR5eKin,
    SceneOMPLPlanner,
    SceneUR5eSpherizedThreeShelf,
    SceneUR5eSpherizedAirbusShopFloor,
    SceneUR5eSpherizedSingleStool,
)
from paper_sequential_planner.experiments.utilio import (
    check_number_Q,
    check_number_E,
    check_num_edges_unique,
    check_num_supercluster_edges,
    gen_joint_trajectory,
    gen_taskspace_tour,
    yaml_write,
    yaml_read,
    plot_joint_trajectory,
    tour_rotation,
    tour_attach_loop_back,
    RTSPLogger,
    dir_logs,
)

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)

problem_dict = [
    ["airbus_shopfloor_RoboTSP_mjd_minits", SceneUR5eSpherizedAirbusShopFloor],
    ["three_shelf_RoboTSP_mjd_minits", SceneUR5eSpherizedThreeShelf],
    ["single_stool_RoboTSP_mjd_minits", SceneUR5eSpherizedSingleStool],
]
problem_selected = 2

PROBLEM_NAME = problem_dict[problem_selected][0]
scene = problem_dict[problem_selected][1](ts_choice="vary24")

robkin = RobotUR5eKin()
planner = SceneOMPLPlanner(scene.collision_check)

unique_sols8 = 8
dof6 = 6
limit6 = np.array(
    [
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-np.pi, np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
    ]
)  # if the table is under, q2 is limited to [-pi, 0]
# limit6[1, 0] = -np.pi
# limit6[1, 1] = 0


# ============= Normal IK and Validity Check =============
def wspace_ik_normal(robot, Xtspace):
    ntasks = Xtspace.shape[0]
    Qaik = np.full((ntasks, unique_sols8, dof6), np.nan)
    Htasks = Xlist_to_Hlist(Xtspace)
    for taski in range(ntasks):
        nik, q_sols = robot.solve_aik(Htasks[taski])
        if nik == 0:
            Qaik[taski] = np.nan
        for qi, q in enumerate(q_sols):
            Qaik[taski, qi] = q
    return Qaik


def batch_Qaik_normal_collision(Qaik, robscene):
    # Qaik shape: (ntasks, num_sols, dof)
    ntasks = Qaik.shape[0]
    Qmin_rep = np.empty((ntasks, unique_sols8, dof6))
    for taski in range(ntasks):
        for solj in range(unique_sols8):
            q = Qaik[taski, solj]  # (dof,)
            Qmin_rep[taski, solj] = q
    Qmin_flat = Qmin_rep.reshape(-1, dof6)  # (ntasks*unique_sols, dof)
    col_states = robscene.collision_check(Qmin_flat).detach().cpu().numpy()
    col_states_rep = col_states.reshape(ntasks, unique_sols8)
    return col_states_rep


def wspace_ik_validity_normal(Qaik, robscene):
    Qaik_rep_col = batch_Qaik_normal_collision(Qaik, robscene)  # batch colcheck
    ntasks = Qaik.shape[0]
    num_sols = Qaik.shape[1]
    eps = 1e-9
    Qaik_valid = np.full((ntasks, num_sols, 1), np.nan)
    for taski in range(ntasks):
        for solj in range(num_sols):
            q = Qaik[taski, solj]
            if np.isnan(q).any():
                Qaik_valid[taski, solj] = -1  # No solution
            else:
                isCollsion = Qaik_rep_col[taski, solj]
                if isCollsion:
                    Qaik_valid[taski, solj] = -2  # In collision
                else:
                    is_in_limit = np.all(
                        (q >= limit6[:, 0] - eps) & (q <= limit6[:, 1] + eps)
                    )
                    if not is_in_limit:
                        Qaik_valid[taski, solj] = -3  # Out of limits
                    else:
                        Qaik_valid[taski, solj] = 1  # Valid solution
    return Qaik_valid.astype(int)


# ============= Normal IK and Validity Check =============


# ============= Extended IK and Validity Check=============
def wspace_ik_extended(robot, Xtspace):
    # this is general from hardware, this has 32 redundant solutions
    alt_num = 32
    ntasks = Xtspace.shape[0]
    num_sols = unique_sols8 * alt_num
    Qaik = np.full((ntasks, num_sols, dof6), np.nan)

    Htasks = Xlist_to_Hlist(Xtspace)
    for taski in range(ntasks):
        nik, q_sols = robot.solve_aik(Htasks[taski])
        if nik == 0:
            Qaik[taski] = np.nan
        for qi, q in enumerate(q_sols):
            q = q + 1e-2  # to avoid numerical issues in find_alt_config2
            alt_qs = find_alt_config2(q, limit6, filterOriginalq=False)
            Qaik[taski, qi * alt_num : (qi + 1) * alt_num] = alt_qs
    return Qaik


def batch_Qaik_extended_collision(Qaik, robscene):
    # Qaik shape: (ntasks, num_sols, dof)
    ntasks = Qaik.shape[0]
    alt_num = 32
    num_sols = unique_sols8 * alt_num
    Qmin_rep = np.empty((ntasks, unique_sols8, dof6))
    for taski in range(ntasks):
        for solj in range(unique_sols8):
            i = solj * alt_num
            j = (solj + 1) * alt_num
            Q = Qaik[taski, i:j]  # (alt_num, dof)
            q = Q[0]
            Qmin_rep[taski, solj] = q
    Qmin_flat = Qmin_rep.reshape(-1, dof6)  # (ntasks*unique_sols, dof)
    col_states = robscene.collision_check(Qmin_flat).detach().cpu().numpy()
    col_states_rep = col_states.reshape(ntasks, unique_sols8)
    Qaik_rep = np.repeat(col_states_rep[:, :, np.newaxis], alt_num, axis=2)
    Qaik_rep_col = Qaik_rep.reshape(ntasks, num_sols)  # (ntasks, num_sols)
    return Qaik_rep_col


def wspace_ik_validity_extended(Qaik, robscene):
    Qaik_rep_col = batch_Qaik_extended_collision(Qaik, robscene)  # batch colcheck
    ntasks = Qaik.shape[0]
    num_sols = Qaik.shape[1]
    eps = 1e-9
    Qaik_valid = np.full((ntasks, num_sols, 1), np.nan)
    for taski in range(ntasks):
        for solj in range(num_sols):
            q = Qaik[taski, solj]
            if np.isnan(q).any():
                Qaik_valid[taski, solj] = -1  # No solution
            else:
                isCollsion = Qaik_rep_col[taski, solj]
                if isCollsion:
                    Qaik_valid[taski, solj] = -2  # In collision
                else:
                    is_in_limit = np.all(
                        (q >= limit6[:, 0] - eps) & (q <= limit6[:, 1] + eps)
                    )
                    if not is_in_limit:
                        Qaik_valid[taski, solj] = -3  # Out of limits
                    else:
                        Qaik_valid[taski, solj] = 1  # Valid solution
    return Qaik_valid.astype(int)


# ============= Extended IK and Validity Check=============

cputtotal_start = time.time()
# preliminary input data processing
qinit = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])
Xinit = H_to_X(robkin.solve_fk(qinit))
H = scene.H
X = Hlist_to_Xlist(H)
ntasks = X.shape[0]
# Qik = wspace_ik_extended(robkin, X)
# Qiks = wspace_ik_validity_extended(Qik, scene)
Qik = wspace_ik_normal(robkin, X)
Qiks = wspace_ik_validity_normal(Qik, scene)
num_sols = Qik.shape[1]

# filter out the unreachable tasks
Xunreach = np.all(Qiks != 1, axis=1).flatten()
X_reach = X[~Xunreach]
Qik_reach = Qik[~Xunreach]
Qiks_reach = Qiks[~Xunreach]

# concat the init to the reachable tasks
qinit_ = np.full((1, Qik_reach.shape[1], Qik_reach.shape[2]), np.nan)
qinit_[0, 0] = qinit
Qik_reach_init = np.vstack((qinit_, Qik_reach))  # init & ntasks
qinit_s_ = np.full((1, Qiks_reach.shape[1], Qiks_reach.shape[2]), -1)
qinit_s_[0, 0] = 1
Qiks_reach_init = np.vstack((qinit_s_, Qiks_reach))  # init & ntasks
Qiks_reach_init = np.where(Qiks_reach_init == 1, True, False)  # T/F mask
X_reach_init = np.vstack((Xinit, X_reach))  # init & ntasks
H_reach_init = Xlist_to_Hlist(X_reach_init)  # init & ntasks

# taskspace relationship analysis
# tmap = Naive_task_space_correlation(H_reach_init)
tmap = KRNN_task_space_correlation(H_reach_init, w_rot=0.0, nnr=0.15, nnk=10)
adjmat = nn_dict_to_adjmat(tmap)
print(f"==>> adjmat: \n{adjmat}")

raise
# taskspace TSP ordering
Ttour = taskspace_tsp_position_distance_order(H_reach_init, tsp_method="local")
Ttour_rotated = tour_rotation(Ttour, start_node=0)
Ttour_rotated_loop = tour_attach_loop_back(Ttour_rotated)
print(f"==>> Ttour_rotated_loop: \n{Ttour_rotated_loop}")

# qdot = np.array([1.57] * 6)  # allowable joint velocity for each joint
qdot = np.array([1.0] * 6)  # allowable joint velocity for each joint
# qw = np.array([10, 10, 10, 1, 1, 0.1])  # move joint 1,2,3 less
qw = np.array([1, 1, 1, 1, 1, 1])  # all joints equal, test exact solver
Wwmj = qw / qdot
Ewmj = Eest_weighted_max_joint_diff(Qik_reach_init, Qiks_reach_init, Wwmj, tmap)

Qtour = Qtour_RoboTSP_layer_search(Ttour_rotated_loop, Ewmj, tmap)
Qtourlist = Qtour_RoboTSP_layer_search_topK(Ttour_rotated_loop, Ewmj, tmap, K=100)
selected = np.vstack([Ttour_rotated_loop, Qtour])
print(f"==>> selected: \n{selected}")

# determine the redundant intervals in the tour
interval_unique = set()
for p in Qtourlist:
    for i in range(len(p) - 1):
        li = i  # layer index -> use given tour to find the task index
        lj = i + 1
        soli = int(p[i])  # solution index of the task at layer li
        solj = int(p[i + 1])
        interval_unique.add((li, lj, soli, solj))
print(f"==>> interval_unique: \n{interval_unique}")

# call expensive collision-free planner and check cost per interval
interval_costs = {}
for interval in tqdm.tqdm(interval_unique):
    # i,j is layer index, we must backtrack to find the task index
    i, j, qi, qj = interval
    # fake cost
    interval_costs[interval] = np.random.rand() * 10
print(f"==>> interval_costs: \n{interval_costs}")

# repropagate the costs back to the Qtourlist
Costlist = np.zeros((Qtourlist.shape[0], Qtourlist.shape[1] - 1))
for pidx, p in enumerate(Qtourlist):
    for i in range(len(p) - 1):
        interval = (i, i + 1, int(p[i]), int(p[i + 1]))
        Costlist[pidx, i] = interval_costs[interval]
print(f"==>> Costlist: \n{Costlist}")
Costlist_total = np.sum(Costlist, axis=1)
print(f"==>> Costlist_total: \n{Costlist_total}")

tourQval = []
for i in range(selected.shape[1]):
    taski = selected[0, i]
    solj = selected[1, i]
    q = Qik_reach_init[taski, solj]
    tourQval.append(q)
tourQval = np.array(tourQval)
print(f"==>> tourQval: \n{tourQval}")

# no collision consider
tourQcosts = traj_complete_cost(tourQval, qdot)
Qtour_traj, time_fs = traj_tour_from_lininterp_qdot(tourQval, qdot)

raise
# collision-free
cputcf_start = time.time()
tourQval_cf = planner.query_tour_planning(tourQval)
tourQcosts_cf = traj_complete_cost(tourQval_cf, qdot)
Qtour_cf_traj, time_fs_cf = traj_tour_from_lininterp_qdot(tourQval_cf, qdot)

cputcf = time.time() - cputcf_start
cputtotal = time.time() - cputtotal_start

raise
# logging
rl = RTSPLogger()
rl.data.pname = PROBLEM_NAME
rl.data.robot = "UR5e"
rl.data.ntasks = ntasks
rl.data.nrtasks = X_reach_init.shape[0]
rl.data.nQpt = num_sols
rl.data.nEpp = num_sols * num_sols
rl.data.ntasks_comb = ntasks * (ntasks - 1) // 2
rl.data.nE_comb = num_sols * (num_sols - 1) // 2
rl.data.tnrQ = np.sum(np.sum(Qiks_reach_init, axis=1)).item()
rl.data.tnrE = None

rl.data.Qf = None
rl.data.Qf_d = None
rl.data.Eest = "Eest_weighted_max_joint_diff"
rl.data.Eest_d = f"Wwmj={Wwmj}"
rl.data.GTSP_svr = None
rl.data.GTSP_svr_d = None

rl.data.l1 = tourQcosts["manhattan"]
rl.data.l2 = tourQcosts["euclidean"]
rl.data.linf = tourQcosts["inf"]
rl.data.time = tourQcosts["time"]
rl.data.l1pj = tourQcosts["perjoint"]

rl.data.l1cf = tourQcosts_cf["manhattan"]
rl.data.l2cf = tourQcosts_cf["euclidean"]
rl.data.linfcf = tourQcosts_cf["inf"]
rl.data.timecf = tourQcosts_cf["time"]
rl.data.l1pjcf = tourQcosts_cf["perjoint"]

rl.data.cputtotal = cputtotal
rl.data.cputQf = None
rl.data.cputEest = None
rl.data.cputGTSP = None
rl.data.cputcf = cputcf
rl.print_log()

# logging to file
# tspace
tsdict = gen_taskspace_tour(X_reach_init, Ttour_rotated)
patht = os.path.join(dir_logs, f"{PROBLEM_NAME}/taskspace_tour.yaml")
yaml_write(patht, tsdict)

# cspace no collision
jtdict = gen_joint_trajectory(Qtour_traj, time_fs, name=PROBLEM_NAME)
pathj = os.path.join(dir_logs, f"{PROBLEM_NAME}/joint_trajectory.yaml")
yaml_write(pathj, jtdict)

# cspace collision-free
jtdict_cf = gen_joint_trajectory(Qtour_cf_traj, time_fs_cf, name=PROBLEM_NAME)
pathj_cf = os.path.join(dir_logs, f"{PROBLEM_NAME}/joint_trajectory_colfree.yaml")
yaml_write(pathj_cf, jtdict_cf)

# data and figure log
logpath = os.path.join(dir_logs, f"{PROBLEM_NAME}/rtsp_log")
rl.save_log(logpath)
plot_joint_trajectory(jtdict, logpath + "_joint_trajectory.png")
plot_joint_trajectory(jtdict_cf, logpath + "_joint_trajectory_colfree.png")
