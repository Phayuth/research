import csv
import numpy as np


def extract_paths(file_path):
    """
    Reads a TSV file and extracts the path for each source-target pair.
    """
    with open(file_path, "r") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")

        paths = []
        for row in reader:
            source = row.get("source")
            target = row.get("target")
            path_str = row.get("path", "")
            if path_str:
                try:
                    nodes = [int(node) for node in path_str.split(",")]
                    print(f"Source: {source}, Target: {target}")
                    print(f"  Path: {nodes}")
                    paths.append(nodes)
                except ValueError as e:
                    print(
                        f"Warning: Could not process path for Source: {source}, Target: {target}. Error: {e}"
                    )
                    paths.append(None)
    return paths


def debug_print_gtsp_():
    pass


def generate_gtsp_header(name, dimension, gtsp_sets):
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


def generate_gtsp_edge_weight_section(
    nodesid_og, num_sols, task_to_nn_pair, Ecspace_eudist, Ecspace_eudist_red
):
    # Extract task and solution IDs for each flat node
    node_tasks = nodesid_og // num_sols
    node_sols = nodesid_og % num_sols
    n_nodes = len(nodesid_og)

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
                is_valid = Ecspace_eudist_red[tp_idx, si, sj]
                if is_valid:
                    gtsp_dist_matrix[i_pos, j_pos] = Ecspace_eudist[tp_idx, si, sj]
            else:
                is_valid = Ecspace_eudist_red[tp_idx, sj, si]
                if is_valid:
                    gtsp_dist_matrix[i_pos, j_pos] = Ecspace_eudist[tp_idx, sj, si]

    gtsp_dist_matrix_int = (gtsp_dist_matrix * 1000).astype(int)

    # Generate GTSP_EDGE_WEIGHT_SECTION
    lines = ["EDGE_WEIGHT_SECTION"]
    for i in range(n_nodes):
        row = gtsp_dist_matrix_int[i]
        row_str = " ".join(str(int(x)) for x in row)
        lines.append(row_str)

    return "\n".join(lines)


def generate_gtsp_set_section(nQredfinalpt, nodesid_cont):
    lines = ["GTSP_SET_SECTION"]
    node_idx = 0
    for task_id, num_nodes in enumerate(nQredfinalpt, start=1):
        # Get the nodes for this task
        task_nodes = nodesid_cont[node_idx : node_idx + num_nodes.item()]
        # Format: task_id node1 node2 ... nodeN -1
        nodes_str = " ".join(map(str, task_nodes))
        lines.append(f"{task_id} {nodes_str} -1")
        node_idx += num_nodes.item()
    lines.append("EOF")
    return "\n".join(lines)


def write_gtsp_file(
    filename="output.gtsp",
    name="GTSP_Instance",
    task_to_nn_pair=None,
    Ecspace_eudist=None,
    Ecspace_eudist_red=None,
    Ecspace_colfree=None,
    Qreduced_final=None
):
    print(f"Writing GTSP file to {filename}...")

    Ecspace_eudist = None  # Placeholder for the actual Ecspace_eudist data
    Ecspace_eudist_red = None  # Placeholder for the actual Ecspace_eudist
    Ecspace_colfree = None  # Placeholder for the actual Ecspace_colfree data

    Qreduced_final = None
    nQredfinalpt = np.sum(Qreduced_final, axis=1)
    print(f"==>> nQredfinalpt: \n{nQredfinalpt.T}")
    nQredfinal = np.sum(nQredfinalpt)
    print(f"==>> nQredfinal: \n{nQredfinal}")

    ntasks, num_sols, dof = Ecspace_eudist.shape
    dimension = nQredfinal
    gtsp_sets = ntasks

    # mapping the flatten node id
    # nodeid_og is the original node id
    # nodeid_cont is the continuous node id for gtsp solver
    Qreduced_final_flat = Qreduced_final.flatten()
    nodesid_og = np.where(Qreduced_final_flat)[0]  # take only the True nodes
    nodesid_cont = np.arange(nQredfinal) + 1  # GTSP node id start from 1
    print(f"==>> nodesid_og.shape: \n{nodesid_og.shape}")
    print(f"==>> nodesid_og: \n{nodesid_og}")
    print(f"==>> nodesid_cont.shape: \n{nodesid_cont.shape}")
    print(f"==>> nodesid_cont: \n{nodesid_cont}")

    header = generate_gtsp_header(
        name,
        dimension,
        gtsp_sets,
    )
    edge_section = generate_gtsp_edge_weight_section(
        nodesid_og,
        num_sols,
        task_to_nn_pair,
        Ecspace_eudist,
        Ecspace_eudist_red,
    )
    set_section = generate_gtsp_set_section(nQredfinalpt, nodesid_cont)

    gtsp_content = f"{header}\n\n{edge_section}\n\n{set_section}"

    with open(filename, "w") as f:
        f.write(gtsp_content)
    print(f"GTSP file written to {filename}")


def read_gtsp_file(input_file):
    pass


if __name__ == "__main__":
    tsv_file = "combined_paths.tsv"
    extract_paths(tsv_file)
