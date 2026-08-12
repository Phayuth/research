import numpy as np
from itertools import permutations


def random_shuffle_sampler(ntasks):
    arr = np.arange(1, ntasks)
    np.random.shuffle(arr)
    arr = np.insert(arr, 0, 0)  # insert start_node=0 at the beginning
    return arr


def generate_permutations(n):
    """
    Generate all unique cyclic permutations with 0 fixed first.

    Returns:
        generator of permutations
    """
    for p in permutations(range(1, n)):
        yield (0,) + p


def nn_dict_to_adjmat(nn_dict):
    ntasks = len(nn_dict)
    adjmat = np.zeros((ntasks, ntasks), dtype=int)

    for task, neighbors in nn_dict.items():
        for neighbor in neighbors:
            adjmat[task, neighbor] = 1

    return adjmat


def forwardsearch_hamiltonian_cycle(adjMat):

    # Check if it's valid to place vertex at current position
    def isSafe(vertex, adjMat, path, pos):

        # The vertex must be adjacent to the previous vertex
        if adjMat[path[pos - 1]][vertex] == 0:
            return False

        # The vertex must not already be in the path
        for i in range(pos):
            if path[i] == vertex:
                return False

        return True

    # Recursive backtracking to construct Hamiltonian Cycle
    def hamCycleUtil(adjMat, path, pos, n):

        # Base case: all vertices are in the path (reached the end)
        if pos == n:
            # Check if there's an edge from last to first vertex
            return adjMat[path[pos - 1]][path[0]] == 1

        # Try all possible vertices as next candidate
        for v in range(1, n):
            if isSafe(v, adjMat, path, pos):
                path[pos] = v
                if hamCycleUtil(adjMat, path, pos + 1, n):
                    return True
                # Backtrack if v doesn't lead to a solution
                path[pos] = -1
        return False

    def hamCycle(adjMat):
        n = len(adjMat)
        path = [-1] * n
        path[0] = 0  # Start path with vertex 0
        if not hamCycleUtil(adjMat, path, 1, n):
            return [-1]
        return path

    return hamCycle(adjMat)


def forwardbacksearch(ntasks, nn):
    start_node = 0

    tour = [None] * ntasks  # the tour suppose to have the length of ntasks
    tour[0] = start_node  # the first node is the start_node
    # tour[-1] = np.random.choice(nn[start_node]) if np.random.choice(nn


if __name__ == "__main__":
    r = random_shuffle_sampler(300)
    print(f"==>> r: \n{r}")

    ntasks = 9
    nn = {
        0: [1, 2, 3, 4, 5, 6, 7, 8],
        1: [0, 2, 3, 4, 5, 6, 7, 8],
        2: [0, 1, 3, 4, 5, 6, 7, 8],
        3: [0, 1, 2, 4, 5, 6, 7, 8],
        4: [0, 1, 2, 3, 5, 6, 7, 8],
        5: [0, 1, 2, 3, 4, 6, 7, 8],
        6: [0, 1, 2, 3, 4, 5, 7, 8],
        7: [0, 1, 2, 3, 4, 5, 6, 8],
        8: [0, 1, 2, 3, 4, 5, 6, 7],
    }

    adjm = nn_dict_to_adjmat(nn)
    print(f"==>> adjm: \n{adjm}")

    path = forwardsearch_hamiltonian_cycle(adjm)
    if path[0] == -1:
        print("Solution does not Exist")
    else:
        print(path)
