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
    Advanced_task_space_correlation,
    position_pairwise_distances,
)
from paper_sequential_planner.scripts.geometric_config import (
    weighted_nan_euclidean_distances,
    weighted_nan_max_joint_diff_distances,
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
    Eest_colfree,
    Eest_weighted_euclidean,
    Eest_weighted_max_joint_diff,
    Qtour_RoboTSP_layer_search,
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
    tsp_solver,
    tour_rotation,
    tour_attach_loop_back,
    RTSPLogger,
)

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
dir_rsrc = os.environ["RSRC_DIR"]
dir_urdf = os.path.join(dir_rsrc, "urdfs")
dir_rtsp = os.path.join(dir_rsrc, "rtsp_env")
dir_glns = os.path.join(dir_rtsp, "gtsp_glns")


PROBLEM_NAME = "three_shelf_RoboTSP_maxjointdiff"
scene = SceneUR5eSpherizedThreeShelf()
# PROBLEM_NAME = "single_stool_Tspaceonly"
# scene = SceneUR5eSpherizedSingleStool()

robkin = RobotUR5eKin()
planner = SceneOMPLPlanner(scene.collision_check)

alt_num = 32
unique_sols = 8
num_sols = unique_sols * alt_num
dof = 6


def wspace_ik_extended(robot, Xtspace):
    # this is general from hardware, this has 32 redundant solutions
    limit6 = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-np.pi, np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )
    # if the table is under, q2 is limited to [-pi, 0]
    # limit6[1, 0] = -np.pi
    # limit6[1, 1] = 0

    ntasks = Xtspace.shape[0]
    Qaik = np.full((ntasks, num_sols, dof), np.nan)

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


def queue_Qaik_batch_collision(Qaik, robscene):
    # Qaik shape: (ntasks, num_sols, dof)
    ntasks = Qaik.shape[0]
    Qmin_rep = np.empty((ntasks, unique_sols, dof))
    for taski in range(ntasks):
        for solj in range(unique_sols):
            i = solj * alt_num
            j = (solj + 1) * alt_num
            Q = Qaik[taski, i:j]  # (alt_num, dof)
            q = Q[0]
            Qmin_rep[taski, solj] = q
    Qmin_flat = Qmin_rep.reshape(-1, dof)  # (ntasks*unique_sols, dof)
    col_states = robscene.collision_check(Qmin_flat).detach().cpu().numpy()
    col_states_rep = col_states.reshape(ntasks, unique_sols)
    Qaik_rep = np.repeat(col_states_rep[:, :, np.newaxis], alt_num, axis=2)
    Qaik_rep_col = Qaik_rep.reshape(ntasks, num_sols)  # (ntasks, num_sols)
    return Qaik_rep_col


def wspace_ik_validity_extended(Qaik, robscene):
    limit6 = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-np.pi, np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )
    Qaik_rep_col = queue_Qaik_batch_collision(Qaik, robscene)  # batch colcheck
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


# preliminary input data processing
qinit = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])
# qinit = np.array([-1.973, -1.094, -1.986, 0.024, 1.701, 0])
Xinit = H_to_X(robkin.solve_fk(qinit))
H = scene.H
X = Hlist_to_Xlist(H)
ntasks = X.shape[0]
Qik = wspace_ik_extended(robkin, X)
Qikstate = wspace_ik_validity_extended(Qik, scene)

# filter out the unreachable tasks
Xunreach = np.all(Qikstate != 1, axis=1).flatten()
X_reach = X[~Xunreach]
Qik_reach = Qik[~Xunreach]
Qikstate_reach = Qikstate[~Xunreach]

# concat the init to the reachable tasks
qinit_ = np.full((1, Qik_reach.shape[1], Qik_reach.shape[2]), np.nan)
qinit_[0, 0] = qinit
Qik_reach_init = np.vstack((qinit_, Qik_reach))  # init & ntasks
qinit_state_ = np.full((1, Qikstate_reach.shape[1], Qikstate_reach.shape[2]), -1)
qinit_state_[0, 0] = 1
Qikstate_reach_init = np.vstack((qinit_state_, Qikstate_reach))  # init & ntasks
Qikstate_reach_init = np.where(Qikstate_reach_init == 1, True, False)  # T/F mask
X_reach_init = np.vstack((Xinit, X_reach))  # init & ntasks
H_reach_init = Xlist_to_Hlist(X_reach_init)  # init & ntasks

# taskspace relationship analysis
tspace_mapping = Naive_task_space_correlation(H_reach_init)

task_to_nn_dict, task_to_nn_pair, task_to_nn_pair_len = (
    tspace_mapping["task_to_nn_dict"],
    tspace_mapping["task_to_nn_pair"],
    tspace_mapping["task_to_nn_pair_len"],
)
print(f"==>> task_to_nn_dict: \n{task_to_nn_dict}")
print(f"==>> task_to_nn_pair: \n{task_to_nn_pair}")
print(f"==>> task_to_nn_pair_len: \n{task_to_nn_pair_len}")

D = position_pairwise_distances(H_reach_init)
Dint = D * 10000
Ttour, cost = tsp_solver(Dint.astype(np.int64), method="local_solver")
print(f"==>> Ttour: {Ttour}")

# taskspace write
Ttour_rotated = tour_rotation(Ttour, start_node=0)
tsdict = gen_taskspace_tour(X_reach_init, Ttour_rotated)
patht = os.path.join(dir_rtsp, f"{PROBLEM_NAME}_taskspace_tour.yaml")
yaml_write(patht, tsdict)


qdot = np.array([1.57] * 6)  # allowable joint velocity for each joint
qw = np.array([10, 10, 10, 1, 1, 0.1])  # move joint 1,2,3 less
Wwmj = qw / qdot
Ewmj = Eest_weighted_max_joint_diff(
    Qik_reach_init, Qikstate_reach_init, Wwmj, tspace_mapping
)
print(f"==>> Ewmj.shape: \n{Ewmj.shape}")

Ttour_rotated_loop = tour_attach_loop_back(Ttour_rotated)
print(f"==>> Ttour_rotated: \n{Ttour_rotated}")
print(f"==>> Ttour_rotated_loop: \n{Ttour_rotated_loop}")
Qtour = Qtour_RoboTSP_layer_search(Ttour_rotated_loop, Ewmj, tspace_mapping)
selected = np.vstack([Ttour_rotated_loop, Qtour])
print(f"==>> selected: \n{selected}")

tourQval = []
for i in range(selected.shape[1]):
    taski = selected[0, i]
    solj = selected[1, i]
    q = Qik_reach_init[taski, solj]
    tourQval.append(q)
tourQval = np.array(tourQval)
print(f"==>> tourQval: \n{tourQval}")

# no collision consider yet
qdot = np.array([1.57] * 6)  # allowable joint velocity for each joint
tourQcost_complete = traj_complete_cost(tourQval, qdot)
Qtour_traj, time_from_start = traj_tour_from_lininterp_qdot(tourQval, qdot)
jtdict = gen_joint_trajectory(Qtour_traj, time_from_start, name=PROBLEM_NAME)
pathj = os.path.join(dir_rtsp, f"{PROBLEM_NAME}_joint_trajectory.yaml")
yaml_write(pathj, jtdict)

# collision-free
tourQval_cf = planner.query_tour_planning(tourQval)
tourQcost_cf_complete = traj_complete_cost(tourQval_cf, qdot)
Qtour_cf_traj, time_from_start_cf = traj_tour_from_lininterp_qdot(
    tourQval_cf, qdot
)
jtdict_cf = gen_joint_trajectory(
    Qtour_cf_traj, time_from_start_cf, name=PROBLEM_NAME
)
pathj_cf = os.path.join(
    dir_rtsp, f"{PROBLEM_NAME}_joint_trajectory_collisionfree.yaml"
)
yaml_write(pathj_cf, jtdict_cf)

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
rl.data.tnrQ = np.sum(np.sum(Qikstate_reach_init, axis=1))
rl.data.tnrE = None

rl.data.Qf = None
rl.data.Qf_d = None
rl.data.Eest = None
rl.data.Eest_d = None
rl.data.GTSP_svr = None
rl.data.GTSP_svr_d = None

rl.data.l1 = tourQcost_complete["manhattan"]
rl.data.l2 = tourQcost_complete["euclidean"]
rl.data.linf = tourQcost_complete["inf"]
rl.data.time = tourQcost_complete["time"]
rl.data.l1pj = tourQcost_complete["perjoint"]

rl.data.l1cf = tourQcost_cf_complete["manhattan"]
rl.data.l2cf = tourQcost_cf_complete["euclidean"]
rl.data.linfcf = tourQcost_cf_complete["inf"]
rl.data.timecf = tourQcost_cf_complete["time"]
rl.data.l1pjcf = tourQcost_cf_complete["perjoint"]
rl.print_log()

logpath = os.path.join(dir_rtsp, f"{PROBLEM_NAME}_rtsp_log")
rl.save_log(logpath)
plot_joint_trajectory(jtdict, savepath=logpath + "_joint_trajectory.png")
plot_joint_trajectory(
    jtdict_cf, savepath=logpath + "_joint_trajectory_collisionfree.png"
)
