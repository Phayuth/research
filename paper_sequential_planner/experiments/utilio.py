import yaml
import numpy as np
import subprocess
import os

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def check_number_Q(Q):
    """
    Q must be state True/False
    """
    nQvalpt = np.sum(Q, axis=1)
    nQvalAll = np.sum(nQvalpt)
    print("------------------------------------------------------------")
    print(f"There are {nQvalAll} valid nodes in total.")
    print(f"Q is in shape {Q.shape}, nQval per task : \n{nQvalpt.T}")
    print("------------------------------------------------------------")


def check_number_E(E):
    """
    E must be state True/False
    """
    nEpair = E.shape[0]
    nEvalpt = np.sum(E, axis=(1, 2))
    nEvalAll = np.sum(nEvalpt)
    print("------------------------------------------------------------")
    print(f"There are {nEpair} pairs with {nEvalAll} valid edges in total.")
    print(f"E is in shape {E.shape}, nEval per pair : \n{nEvalpt}")
    print("------------------------------------------------------------")


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


def write_gtsp_file(filename, instancename, task_to_nn_pair, E, Q):
    print(f"==>> Writing GTSP file to {filename} !")

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

    header = gen_gtsp_header(instancename, dimension, gtsp_sets)
    ed = gen_gtsp_ew_section(Qid_true, num_sols, task_to_nn_pair, E)
    set = gen_gtsp_set_section(nQpt, Qid_true_cont)
    gtsp = f"{header}\n\n{ed}\n\n{set}"
    with open(filename, "w") as f:
        f.write(gtsp)

    print(f"==>> GTSP file written to {filename} !")

    return Qid_true, Qid_true_cont


def read_gtsp_file(filename, Qid_true, Qid_true_cont):
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
    tour_indices_og_rotated = rotate_tour(tour_indices_og, start_node=0)

    # print debug info
    print("------------------------------------------------------------")
    print(f"GLNS Tour IDs (flattened): {tour_glns}")
    print(f"Tour indices in original node IDs: {tour_indices_og}")
    print(f"Rotated tour indices: {tour_indices_og_rotated}")

    return tour_indices_og_rotated


def rotate_tour(tour_indices_og, start_node):
    """
    Rotate the tour so that it starts with the specified start_node.
    """
    if start_node not in tour_indices_og:
        raise ValueError(f"Start node {start_node} not found in the tour.")

    start_index = np.where(tour_indices_og == start_node)[0][0]
    rotated_tour = np.concatenate(
        (tour_indices_og[start_index:], tour_indices_og[:start_index])
    )
    return rotated_tour


def call_gtsp_glns_solver(
    solver_dir, input_file, output_file=None, args=None, check=True, verbose=True
):
    """
    Generic caller for external command-line solvers.

    Args:
        solver_dir: Directory containing the solver executable
        input_file: Path to input file
        output_file: Path to output file (optional)
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
    ./GLNScmd.jl ur5e_sphere_gtsp.gtsp -output=ur5e_sols
    """

    # Build command list
    solver_cmd = "GLNScmd.jl"
    solver_path = os.path.join(solver_dir, solver_cmd)
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


def yaml_write(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def yaml_read(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def gen_joint_trajectory(traj):
    traj_dict = {}
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    traj_dict["joint_names"] = joint_names
    traj_dict["N"] = traj.shape[0]
    traj_dict["points"] = traj.tolist()
    traj_dict["time_from_start"] = (np.arange(traj.shape[0]) * 0.1).tolist()
    return traj_dict


def gen_taskspace_tour(X, Ttour):
    ts_dict = {}
    ts_dict["standard"] = "xyz_qxqyqzqw"
    ts_dict["is_points_ordered"] = False
    ts_dict["order"] = Ttour.tolist()
    ts_dict["points"] = X.tolist()
    ts_dict["N"] = X.shape[0]
    return ts_dict


if __name__ == "__main__":
    pass
