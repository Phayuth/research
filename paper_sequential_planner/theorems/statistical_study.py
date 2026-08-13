import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
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
)
from paper_sequential_planner.scripts.geometric_config import (
    traj_complete_cost,
    traj_tour_from_lininterp_qdot,
)
from paper_sequential_planner.scripts.geometric_rtsp import (
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
    gen_joint_trajectory,
    gen_taskspace_tour,
    yaml_write,
    yaml_read,
    txt_write,
    tour_rotation,
    tour_attach_loop_back,
    dir_logs,
)
from itertools import permutations
from prettytable import PrettyTable

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# problem_dict = [
#     ["airbus_shopfloor_RoboTSP_mjd_minits", SceneUR5eSpherizedAirbusShopFloor],
#     ["three_shelf_RoboTSP_mjd_minits", SceneUR5eSpherizedThreeShelf],
#     ["single_stool_RoboTSP_mjd_minits", SceneUR5eSpherizedSingleStool],
# ]
# problem_selected = 2

PROBLEM_NAME = "single_stool_RoboTSP_mjd_minits"
scene = SceneUR5eSpherizedSingleStool(ts_choice="mini")

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


# preliminary input data processing
qinit = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])
Xinit = H_to_X(robkin.solve_fk(qinit))
H = scene.H
X = Hlist_to_Xlist(H)
ntasks = X.shape[0]
Qik = wspace_ik_extended(robkin, X)
Qikstate = wspace_ik_validity_extended(Qik, scene)
# Qik = wspace_ik_normal(robkin, X)
# Qikstate = wspace_ik_validity_normal(Qik, scene)
num_sols = Qik.shape[1]

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
# tspace_mapping = KRNN_task_space_correlation(
#     H_reach_init,
#     w_rot=0.0,
#     nnr=0.15,
#     nnk=10,
# )
task_to_nn_dict, task_to_nn_pair, task_to_nn_pair_len = (
    tspace_mapping["task_to_nn_dict"],
    tspace_mapping["task_to_nn_pair"],
    tspace_mapping["task_to_nn_pair_len"],
)
print(f"==>> task_to_nn_dict: \n{task_to_nn_dict}")
print(f"==>> task_to_nn_pair: \n{task_to_nn_pair}")
print(f"==>> task_to_nn_pair_len: \n{task_to_nn_pair_len}")

Ttour = taskspace_tsp_position_distance_order(
    H_reach_init,
    tsp_method="local",
)
print(f"==>> Ttour: \n{Ttour}")

# taskspace write
Ttour_rotated = tour_rotation(Ttour, start_node=0)
tsdict = gen_taskspace_tour(X_reach_init, Ttour_rotated)

# qdot = np.array([1.57] * 6)  # allowable joint velocity for each joint
qdot = np.array([1.0] * 6)  # allowable joint velocity for each joint
# qw = np.array([10, 10, 10, 1, 1, 0.1])  # move joint 1,2,3 less
qw = np.array([1, 1, 1, 1, 1, 1])  # all joints equal, test exact solver
Wwmj = qw / qdot
Ewmj = Eest_weighted_max_joint_diff(
    Qik_reach_init, Qikstate_reach_init, Wwmj, tspace_mapping
)

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

# no collision consider
tourQcosts = traj_complete_cost(tourQval, qdot)
Qtour_traj, time_fs = traj_tour_from_lininterp_qdot(tourQval, qdot)
jtdict = gen_joint_trajectory(Qtour_traj, time_fs, name=PROBLEM_NAME)
roboTSP_best_timecost = tourQcosts["time"]
print(f"==>> roboTSP_best_timecost: \n{roboTSP_best_timecost}")


def full_permutation_search():
    print(f"====The full permutation search for global optimality=======")
    perms = taskspace_brute_permutation_order(H_reach_init)
    perms_idx = list(range(len(perms)))

    dict_data = {
        "metadata": {},
        "data": {},
    }
    dict_data["metadata"]["problem_name"] = PROBLEM_NAME
    dict_data["metadata"]["number_of_tasks"] = ntasks
    dict_data["metadata"]["number_of_permutations"] = len(perms)

    best_tour = None
    best_cost = np.inf
    pbar = tqdm.tqdm(
        zip(perms_idx, perms), total=len(perms), desc="Proc permutations"
    )

    for i, p in pbar:
        Ttour = list(p)
        Ttour_rotated = tour_rotation(Ttour, start_node=0)
        Ttour_rotated_loop = tour_attach_loop_back(Ttour_rotated)
        Qtour = Qtour_RoboTSP_layer_search(
            Ttour_rotated_loop, Ewmj, tspace_mapping
        )
        selected = np.vstack([Ttour_rotated_loop, Qtour])

        tourQval = []
        for j in range(selected.shape[1]):
            taski = selected[0, j]
            solj = selected[1, j]
            q = Qik_reach_init[taski, solj]
            tourQval.append(q)
        tourQval = np.array(tourQval)
        tourQcosts = traj_complete_cost(tourQval, qdot)

        dict_data["data"][f"idx_{i}"] = {}
        dict_data["data"][f"idx_{i}"]["id"] = i
        dict_data["data"][f"idx_{i}"]["permutation"] = list(p)
        dict_data["data"][f"idx_{i}"]["cost"] = tourQcosts

        if tourQcosts["time"] < best_cost:
            best_cost = tourQcosts["time"]
            best_tour = p

        pbar.set_postfix(
            {"prm": p, "cost": tourQcosts["time"], "best_cost": best_cost}
        )

    print(f"==>> best_tour: \n{best_tour}")
    print(f"==>> best_cost: \n{best_cost}")
    pathperm = os.path.join(dir_logs, f"{PROBLEM_NAME}/permutations.yaml")
    yaml_write(pathperm, dict_data)


def analyze_permutation_results():
    PROBLEM_NAME = "single_stool_RoboTSP_mjd_minits"
    pathperm = os.path.join(dir_logs, f"{PROBLEM_NAME}/permutations.yaml")
    data_dict = yaml_read(pathperm)

    timecost = [
        data_dict["data"][f"idx_{i}"]["cost"]["time"]
        for i in range(len(data_dict["data"]))
    ]

    # time cost graphs per permutation
    fig, ax = plt.subplots()
    ax.plot(timecost, marker="o", linestyle="-")
    ax.set_xlabel("Permutation Index")
    ax.set_ylabel("Time Cost")
    ax.set_title(f"Time Cost for Permutations - {PROBLEM_NAME}")
    ax.set_xlim(0, len(timecost) - 1)

    # histogram of time costs
    fig2, ax2 = plt.subplots()
    ax2.hist(timecost, bins=100, edgecolor="black")
    ax2.set_xlabel("Time Cost")
    ax2.set_ylabel("Number of permutations")
    ax2.set_title("Distribution of Time Costs")

    # CDF of time costs
    sorted_costs = np.sort(timecost)
    cdf = np.arange(1, len(timecost) + 1) / len(timecost)
    fig3, ax3 = plt.subplots()
    ax3.plot(sorted_costs, cdf)
    ax3.set_xlabel("Time Cost")
    ax3.set_ylabel("P(Time Cost ≤ x)")
    ax3.set_title("CDF of Time Costs")
    ax3.grid()

    # CDF of time costs (10th percentile)
    fig4, ax4 = plt.subplots()
    ax4.plot(sorted_costs, cdf)
    ax4.set_xlim(sorted_costs[0], np.percentile(sorted_costs, 10))
    ax4.set_xlabel("Time Cost")
    ax4.set_ylabel("P(Time Cost ≤ x)")
    ax4.set_title("CDF of Time Costs (10th Percentile)")
    ax4.grid()

    plt.show()

    args = np.argsort(timecost)
    tb = PrettyTable()
    tb.title = f"Sorted Time Costs"
    tb.field_names = ["Permutation Index", "Permutation", "Time Cost"]
    for i in args:
        perm = data_dict["data"][f"idx_{i}"]["permutation"]
        cost = data_dict["data"][f"idx_{i}"]["cost"]["time"]
        tb.add_row([i, perm, cost])
    print(tb)

    # pathtb = os.path.join(dir_logs, f"{PROBLEM_NAME}/permutations_table.txt")
    # txt_write(pathtb, str(tb))

    true_optimum = np.min(timecost)

    sample_ratios = [
        0.001,
        0.005,
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]
    trials = 1000

    for ratio in sample_ratios:
        sample_size = int(ratio * len(timecost))  # how many samples to draw
        bests = []
        for _ in range(trials):
            # randomly sample given size 1000 times and record the best time cost
            sample = np.random.choice(timecost, sample_size, replace=False)
            bests.append(sample.min())
        bests = np.array(bests)

        print(
            f"{ratio*100:.1f}%: "
            f"mean={bests.mean():.3f}, "
            f"best={bests.min():.3f}, "
            f"worst={bests.max():.3f}"
        )


if __name__ == "__main__":
    # full_permutation_search()
    analyze_permutation_results()
