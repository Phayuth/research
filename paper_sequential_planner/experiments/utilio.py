import os
import yaml
import numpy as np
import subprocess
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from datetime import datetime
import fast_tsp
from python_tsp import exact as pytsp_exact, heuristics as pytsp_heuristics

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
dir_rsrc = os.environ["RSRC_DIR"]
dir_urdf = os.path.join(dir_rsrc, "urdfs")
dir_rtsp = os.path.join(dir_rsrc, "rtsp_env")
dir_glns = os.path.join(dir_rtsp, "gtsp_glns")


def check_number_Q(Qs):
    """
    Q must be state True/False
    """
    if not np.issubdtype(Qs.dtype, np.bool_):
        raise ValueError("Q must be a boolean array")

    nQvalpt = np.sum(Qs, axis=1)
    nQvalAll = np.sum(nQvalpt)
    print("------------------------------------------------------------")
    print(f"There are {nQvalAll} valid nodes in total.")
    print(f"Q is in shape {Qs.shape}, nQval per task : \n{nQvalpt.T}")
    print("------------------------------------------------------------")


def check_number_E(Es):
    """
    E must be state True/False
    """
    if not np.issubdtype(Es.dtype, np.bool_):
        raise ValueError("E must be a boolean array")

    nEpair = Es.shape[0]
    nEvalpt = np.sum(Es, axis=(1, 2))
    nEvalAll = np.sum(nEvalpt)
    print("------------------------------------------------------------")
    print(f"There are {nEpair} pairs with {nEvalAll} valid edges in total.")
    print(f"E is in shape {Es.shape}, nEval per pair : \n{nEvalpt}")
    print("------------------------------------------------------------")


def check_num_edges_unique(num_qreachable):
    totalnode = sum(num_qreachable)
    self_connections = sum([n * (n - 1) / 2 for n in num_qreachable])
    unique_edges = (totalnode * (totalnode - 1) / 2) - self_connections
    print("------------------------------------------------------------")
    print(f"Total unique edges: {unique_edges}")
    print("------------------------------------------------------------")
    return unique_edges


def check_num_supercluster_edges(n_tasks):
    supercluster_edges = n_tasks * (n_tasks - 1) / 2
    print("------------------------------------------------------------")
    print(f"Total supercluster edges: {supercluster_edges}")
    print("------------------------------------------------------------")
    return supercluster_edges


def gen_gtsp_header(name, dimension, gtsp_sets):
    """
    DIMENSION is the total sum of every nodes in every cluster (solutions)
    GTSP_SETS is the total number of clusters (tasks)
    """
    lines = []
    lines.append(f"NAME: {name}")
    lines.append("TYPE: GTSP")
    lines.append(f"DIMENSION: {dimension}")
    lines.append(f"GTSP_SETS: {gtsp_sets}")
    lines.append("EDGE_WEIGHT_TYPE: EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT: FULL_MATRIX")
    return "\n".join(lines)


def gen_gtsp_ew_section(Qid_true, num_sols, task_to_nn_pair, Ecost):
    # Extract task and solution IDs for each flat node
    node_tasks = Qid_true // num_sols
    node_sols = Qid_true % num_sols
    n_nodes = len(Qid_true)

    # Build lookup table for task pair indices: task_pair_lookup[i,j] -> task_pair_idx
    task_to_nn_pair_arr = np.array(task_to_nn_pair)
    num_tasks = task_to_nn_pair_arr.max() + 1
    task_pair_lookup = np.full((num_tasks, num_tasks), -1, dtype=np.int32)
    for idx, (i, j) in enumerate(task_to_nn_pair_arr):
        task_pair_lookup[i, j] = idx
        task_pair_lookup[j, i] = idx

    # Create meshgrid for all node pairs
    i_idx, j_idx = np.meshgrid(
        np.arange(n_nodes), np.arange(n_nodes), indexing="ij"
    )
    i_idx_flat = i_idx.ravel()
    j_idx_flat = j_idx.ravel()

    # Get task/sol for each node in all pairs
    task_i = node_tasks[i_idx_flat]
    sol_i = node_sols[i_idx_flat]
    task_j = node_tasks[j_idx_flat]
    sol_j = node_sols[j_idx_flat]

    # Initialize distance matrix
    gtsp_dist_matrix = np.full((n_nodes, n_nodes), 1000, dtype=np.float64)
    np.fill_diagonal(gtsp_dist_matrix, 0)  # Set diagonal to 0

    # Find task pair indices using lookup table
    task_pair_idx = task_pair_lookup[task_i, task_j]

    # Valid pairs: same task pair exists and tasks are different
    valid_pairs = (task_pair_idx >= 0) & (task_i != task_j)

    if np.any(valid_pairs):
        task_pair_idx_valid = task_pair_idx[valid_pairs]
        i_idx_valid = i_idx_flat[valid_pairs]
        j_idx_valid = j_idx_flat[valid_pairs]
        task_i_valid = task_i[valid_pairs]
        sol_i_valid = sol_i[valid_pairs]
        task_j_valid = task_j[valid_pairs]
        sol_j_valid = sol_j[valid_pairs]

        # For each valid pair, check if edge is valid in cspace_eudist_val_selected
        for idx in range(len(task_pair_idx_valid)):
            tp_idx = task_pair_idx_valid[idx]
            ti = task_i_valid[idx]
            tj = task_j_valid[idx]
            si = sol_i_valid[idx]
            sj = sol_j_valid[idx]
            i_pos = i_idx_valid[idx]
            j_pos = j_idx_valid[idx]

            if ti < tj:
                gtsp_dist_matrix[i_pos, j_pos] = Ecost[tp_idx, si, sj]
            else:
                gtsp_dist_matrix[i_pos, j_pos] = Ecost[tp_idx, sj, si]

    gtsp_dist_matrix_int = (gtsp_dist_matrix * 1000).astype(int)

    # Generate GTSP_EDGE_WEIGHT_SECTION
    lines = ["EDGE_WEIGHT_SECTION"]
    for i in range(n_nodes):
        row = gtsp_dist_matrix_int[i]
        row_str = " ".join(str(int(x)) for x in row)
        lines.append(row_str)

    return "\n".join(lines)


def gen_gtsp_set_section(nQredfinalpt, Qid_true_cont):
    lines = ["GTSP_SET_SECTION"]
    node_idx = 0
    for task_id, num_nodes in enumerate(nQredfinalpt, start=1):
        # Get the nodes for this task
        task_nodes = Qid_true_cont[node_idx : node_idx + num_nodes.item()]
        # Format: task_id node1 node2 ... nodeN -1
        nodes_str = " ".join(map(str, task_nodes))
        lines.append(f"{task_id} {nodes_str} -1")
        node_idx += num_nodes.item()
    lines.append("EOF")
    return "\n".join(lines)


def write_glns_file(problem_name, task_to_nn_pair, E, Q):
    problem_path = os.path.join(dir_glns, f"{problem_name}.gtsp")
    print(f"==>> Writing GLNS file to {problem_path} !")

    # determine the number of dimensions and gtsp sets
    nQpt = np.sum(Q, axis=1)
    nQ = np.sum(Q)
    ntasks, num_sols, dof = Q.shape
    dimension = nQ
    gtsp_sets = ntasks

    # mapping the flatten node id
    # nodeid_og is the original node id
    # nodeid_cont is the continuous node id for gtsp solver
    Qid_true = np.where(Q.flatten())[0]  # take only the True nodes
    Qid_true_cont = np.arange(Qid_true.shape[0]) + 1  # GTSP node id start from 1

    head = gen_gtsp_header(problem_name, dimension, gtsp_sets)
    ed = gen_gtsp_ew_section(Qid_true, num_sols, task_to_nn_pair, E)
    set = gen_gtsp_set_section(nQpt, Qid_true_cont)
    gtsp = f"{head}\n\n{ed}\n\n{set}"
    with open(problem_path, "w") as f:
        f.write(gtsp)

    print(f"==>> GTSP file written to {problem_path} !")
    return Qid_true, Qid_true_cont


def call_glns_solver(
    problem_name,
    args=None,
    check=True,
    verbose=True,
):
    """
    Generic caller for external command-line solvers.

    Args:
        solver_dir: Directory containing the solver executable
        args: Dict of additional arguments (e.g., {'mode': 'fast', 'max_time': 60})
        check: Raise exception on non-zero exit code (default: True)
        verbose: Print output messages (default: True)

    Returns:
        CompletedProcess object containing returncode, stdout, stderr

    option
        -max_time=[Int]				 (default set by mode)
        -trials=[Int]				 (default set by mode)
        -restarts=[Int]              (default set by mode)
        -mode=[default, fast, slow]  (default is default)
        -verbose=[0, 1, 2, 3]        (default is 3. 0 is no output, 3 is most verbose)
        -output=[filename]           (default is None)
        -epsilon=[Float in [0,1]]	 (default is 0.5)
        -reopt=[Float in [0,1]]      (default is set by mode)

    terminal command example:
    ./GLNScmd.jl ur5e_sphere_gtsp.gtsp -output=ur5e_sphere_gtsp.sols
    """
    solver_cmd = "GLNScmd.jl"
    solver_dir = dir_glns
    solver_path = os.path.join(solver_dir, solver_cmd)
    input_file = os.path.join(solver_dir, f"{problem_name}.gtsp")
    output_file = os.path.join(solver_dir, f"{problem_name}.sols")
    command = [solver_path, input_file]

    # Add output file if provided
    if output_file:
        command.append(f"-output={output_file}")

    # Add additional arguments
    if args:
        for key, val in args.items():
            command.append(f"-{key}={val}")

    try:
        if verbose:
            print(f"Executing: {' '.join(command)}")

        result = subprocess.run(
            command, check=check, capture_output=True, text=True
        )

        if verbose:
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            print(f"Solver executed successfully (exit code: {result.returncode})")

        return result

    except subprocess.CalledProcessError as e:
        print(f"Error executing solver: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        raise


def read_glns_file(problem_name, Qid_true, Qid_true_cont):
    filename = os.path.join(dir_glns, f"{problem_name}.sols")
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("Tour") and not line.startswith("Tour Cost"):
                tour_str = line.split(":", 1)[1].strip()
                tour_glns = eval(tour_str)
                break

    # remap the tour flatten node id back to its original id
    tour_glns = np.array(tour_glns)
    tour_indices = np.searchsorted(Qid_true_cont, tour_glns)
    tour_indices_og = Qid_true[tour_indices]
    tour_indices_og_rotated = tour_rotation(tour_indices_og, start_node=0)

    # print debug info
    print("------------------------------------------------------------")
    print(f"GLNS Tour IDs (flattened): {tour_glns}")
    print(f"Tour indices in original node IDs: {tour_indices_og}")
    print(f"Rotated tour indices: {tour_indices_og_rotated}")

    return tour_indices_og_rotated


def write_tsp_file(problem_name, E, Q):
    pass


def call_tsp_solver():
    pass


def read_tsp_file():
    pass


def tsp_solver(dists, method):
    if method == "local_solver":
        tour = fast_tsp.find_tour(dists)
        cost = fast_tsp.compute_cost(tour, dists)
    elif method == "greedy_nearest_neighbor":
        tour = fast_tsp.greedy_nearest_neighbor(dists)
        cost = fast_tsp.compute_cost(tour, dists)
    elif method == "exact_held_karp":
        tour = fast_tsp.solve_tsp_exact(dists)
        cost = fast_tsp.compute_cost(tour, dists)

    elif method == "exact_brute_force":
        tour, cost = pytsp_exact.solve_tsp_brute_force(dists)
    elif method == "exact_dynamic_programming":
        tour, cost = pytsp_exact.solve_tsp_dynamic_programming(dists)
    elif method == "exact_branch_and_bound":
        tour, cost = pytsp_exact.solve_tsp_branch_and_bound(dists)

    elif method == "heuristic_local_search":
        tour, cost = pytsp_heuristics.solve_tsp_local_search(dists)
    elif method == "heuristic_simulated_annealing":
        tour, cost = pytsp_heuristics.solve_tsp_simulated_annealing(dists)
    elif method == "heuristic_lin_kernighan":
        tour, cost = pytsp_heuristics.solve_tsp_lin_kernighan(dists)
    elif method == "heuristic_record_to_record":
        tour, cost = pytsp_heuristics.solve_tsp_record_to_record(dists)

    return tour, cost


def tour_rotation(tour_indices_og, start_node):
    """
    Rotate the tour so that it starts with the specified start_node.
    """
    tour_indices_og = np.asarray(tour_indices_og)
    if start_node not in tour_indices_og:
        raise ValueError(f"Start node {start_node} not found in the tour.")

    start_index = np.where(tour_indices_og == start_node)[0][0]
    rotated_tour = np.concatenate(
        (tour_indices_og[start_index:], tour_indices_og[:start_index])
    )
    return rotated_tour


def tour_attach_loop_back(tour):
    """
    Attach the last node back to the first node to form a loop.
    """
    if len(tour) < 2:
        raise ValueError(
            "Tour must have at least two nodes to attach a loop back."
        )
    return np.append(tour, tour[0])


def yaml_write(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def yaml_read(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def gen_joint_trajectory(traj, time_from_start, name):
    traj_dict = {}
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    traj_dict["name"] = name
    traj_dict["joint_names"] = joint_names
    traj_dict["N"] = traj.shape[0]
    traj_dict["points"] = traj.tolist()
    traj_dict["time_from_start"] = time_from_start.tolist()
    return traj_dict


def gen_taskspace_tour(X, Ttour):
    if isinstance(Ttour, list):
        Ttour = np.array(Ttour)

    ts_dict = {}
    ts_dict["standard"] = "xyz_qxqyqzqw"
    ts_dict["is_points_ordered"] = False
    ts_dict["N"] = X.shape[0]
    ts_dict["order"] = Ttour.tolist()
    ts_dict["points"] = X.tolist()
    return ts_dict


def plot_joint_trajectory(traj_dict, savepath=None):
    name = traj_dict["name"]
    joint_names = traj_dict["joint_names"]
    points = np.array(traj_dict["points"])
    time_from_start = np.array(traj_dict["time_from_start"])
    dof = points.shape[1]

    fig, axes = plt.subplots(dof, 1, figsize=(10, 2 * dof))
    fig.suptitle(f"Problem: {name}", fontsize=16, fontweight="bold")
    fig.canvas.manager.set_window_title(f"Problem: {name}")

    markhlines = [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]
    if dof == 1:
        axes = [axes]
    for i in range(dof):
        axes[i].plot(time_from_start, points[:, i])

        for hline in markhlines:
            axes[i].hlines(
                hline,
                time_from_start[0],
                time_from_start[-1],
                colors="r",
                linestyles="dashed",
                alpha=0.3,
            )
        axes[i].set_xlim(time_from_start[0], time_from_start[-1])
        axes[i].set_ylim(-2 * np.pi, 2 * np.pi)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel(f"{joint_names[i]} (rad)")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    else:
        plt.show()


@dataclass
class RTSPLog:
    # precompute data
    d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # metadata
    date: str = field(
        default=d,
        metadata={
            "label": "Date",
            "info": "auto-generated",
        },
    )
    pname: str = field(
        default=None,
        metadata={
            "label": "Problem Name",
            "info": "",
        },
    )
    robot: str = field(
        default=None,
        metadata={
            "label": "Robot",
            "info": "",
        },
    )
    ntasks: int = field(
        default=None,
        metadata={
            "label": "Number of Tasks",
            "info": "user defined",
        },
    )
    nQpt: int = field(
        default=None,
        metadata={
            "label": "Number of Q per Task",
            "info": "",
        },
    )
    nEpp: int = field(
        default=None,
        metadata={
            "label": "Number of E per Pair",
            "info": "",
        },
    )
    ntasks_comb: int = field(
        default=None,
        metadata={
            "label": "Number of Task Comibinations",
            "info": "Unique Task combinations",
        },
    )
    nE_comb: int = field(
        default=None,
        metadata={
            "label": "Number of Edge Combinations",
            "info": "Unique edge combinations",
        },
    )
    nrtasks: int = field(
        default=None,
        metadata={
            "label": "Number of Reachable Tasks",
            "info": "includes initial task",
        },
    )
    tnrQ: int = field(
        default=None,
        metadata={
            "label": "Total Number of Reachable Q",
            "info": "(before filtering)",
        },
    )
    tnrE: int = field(
        default=None,
        metadata={
            "label": "Total Number of Reachable E",
            "info": "(before filtering)",
        },
    )

    # methods
    Qf: str = field(default=None, metadata={"label": "Qfilter"})
    Qf_d: str = field(default=None, metadata={"label": "Qfilter Data"})
    Eest: str = field(default=None, metadata={"label": "Eestimation"})
    Eest_d: str = field(default=None, metadata={"label": "Eestimation Data"})
    GTSP_svr: str = field(default=None, metadata={"label": "GTSP Solver"})
    GTSP_svr_d: str = field(default=None, metadata={"label": "GTSP Solver Data"})

    # results
    l1: float = field(
        default=None, metadata={"label": "Manhattan Cost", "info": "unit:rad"}
    )
    l2: float = field(
        default=None, metadata={"label": "Euclidean Cost", "info": "unit:rad"}
    )
    linf: float = field(
        default=None, metadata={"label": "Infinity Cost", "info": "unit:rad"}
    )
    time: float = field(
        default=None, metadata={"label": "Time Cost", "info": "unit:s"}
    )
    l1pj: float = field(
        default=None, metadata={"label": "Per-Joint Cost", "info": "unit:rad"}
    )

    # collision-free planning
    cf_svr: str = field(default=None, metadata={"label": "Collision-Free Planner"})
    cf_svr_d: str = field(
        default=None, metadata={"label": "Collision-Free Planner Data"}
    )
    l1cf: float = field(
        default=None,
        metadata={"label": "Collision-Free Manhattan Cost", "info": "unit:rad"},
    )
    l2cf: float = field(
        default=None,
        metadata={"label": "Collision-Free Euclidean Cost", "info": "unit:rad"},
    )
    linfcf: float = field(
        default=None,
        metadata={"label": "Collision-Free Infinity Cost", "info": "unit:rad"},
    )
    timecf: float = field(
        default=None,
        metadata={"label": "Collision-Free Time Cost", "info": "unit:s"},
    )
    l1pjcf: float = field(
        default=None,
        metadata={"label": "Collision-Free Per-Joint Cost", "info": "unit:rad"},
    )

    # cpu compute time
    Qft: float = field(
        default=None, metadata={"label": "Qfilter Time", "info": "unit:s"}
    )
    Eestt: float = field(
        default=None, metadata={"label": "Eestimation Time", "info": "unit:s"}
    )
    GTSPt: float = field(
        default=None, metadata={"label": "GTSP Solver Time", "info": "unit:s"}
    )
    cft: float = field(
        default=None,
        metadata={"label": "Collision-Free Planner Time", "info": "unit:s"},
    )


class RTSPLogger:

    def __init__(self):
        self.data = RTSPLog()
        self.tb = PrettyTable()

    def _fmt(self, x):
        if isinstance(x, float):
            return f"{x:.5f}"
        elif isinstance(x, list):
            return [float(f"{v:.5f}") if isinstance(v, float) else v for v in x]
        return x

    def print_log(self):
        self.tb.align = "l"
        self.tb.title = f"RTSP Problem: {self.data.pname}"
        self.tb.field_names = ["Parameter", "Value", "Info"]
        tbd = []
        for f in fields(self.data):
            lb = f.metadata.get("label", f.name)
            val = self._fmt(getattr(self.data, f.name))
            info = f.metadata.get("info", "")
            row = [lb, val, info]
            tbd.append(row)
        self.tb.add_rows(tbd)
        print(self.tb)

    def save_log(self, path):
        # human-readable table
        with open(path + ".txt", "w") as f:
            f.write(str(self.tb))

        # yaml table
        yaml_write(path + ".yaml", self.data.__dict__)


if __name__ == "__main__":
    PROBLEM_NAME = "three_shelf_maxjointdiff_ww_newstart"
    pathj = os.path.join(dir_rtsp, f"{PROBLEM_NAME}_joint_trajectory.yaml")
    jtdict = yaml_read(pathj)
    plot_joint_trajectory(jtdict)
