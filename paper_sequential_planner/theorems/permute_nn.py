import numpy as np
from itertools import permutations, product
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def _nn_dict_to_adjmat(nn_dict):
    ntasks = len(nn_dict)
    adjmat = np.zeros((ntasks, ntasks), dtype=int)

    for task, neighbors in nn_dict.items():
        for neighbor in neighbors:
            adjmat[task, neighbor] = 1

    return adjmat


def generate_random_shuffle(n):
    arr = np.arange(1, n)
    np.random.shuffle(arr)
    arr = np.insert(arr, 0, 0)  # insert start_node=0 at the beginning
    return arr


def generate_block_permutations_first(n, blocks):
    others = [i for i in range(1, n)]
    for b in blocks:  # remove elements in blocks from others
        others.remove(b)

    for f in permutations(blocks):
        for o in permutations(others):  # fresh generator per f
            yield (0,) + f + o


def generate_brute_force_permutations(n):
    """
    Generate all unique cyclic permutations with 0 fixed first.
    Clockwise and counter-clockwise are considered the same.

    Lazy generator, so we have to do something like:
    p = generate_permutations(n)
    while True:
        try:
            s = next(p)
            print(s)
        except StopIteration:
            break

    Returns:
        generator of permutations
    """
    for p in permutations(range(1, n)):
        if p[0] < p[-1]:  # Remove the reverse-equivalent tour
            yield (0,) + p


def generate_spectral_graph_permutations(edges, n, start=0):
    """
    Generate an ordering of n elements using spectral graph ordering.

    Parameters
    ----------
    edges : list of tuple
        KNN edges, e.g. [(0,1), (0,2), (1,3), ...]
    n : int
        Total number of elements.
    start : int
        Element to place first.

    Returns
    -------
    order : list
        Permutation of [0, ..., n-1].
    """

    # build matrix
    rows = []
    cols = []
    for i, j in edges:
        rows.extend([i, j])
        cols.extend([j, i])
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Graph Laplacian L = D - A
    degree = np.asarray(A.sum(axis=1)).ravel()
    D = coo_matrix((degree, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()

    L = D - A

    # Second-smallest eigenvector (Fiedler vector)
    _, eigenvectors = eigsh(L, k=2, which="SM")

    fiedler = eigenvectors[:, 1]

    # Sort nodes according to Fiedler values
    order = np.argsort(fiedler).tolist()

    # Rotate so that `start` is first
    idx = order.index(start)
    order = order[idx:] + order[:idx]

    return order


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


def forwardbacksearch(n, nn):
    start_node = 0

    tour = [None] * n  # the tour suppose to have the length of ntasks
    tour[0] = start_node  # the first node is the start_node
    tour[-1] = np.random.choice(nn[start_node])


def generate_block_permutations(n, blocks):
    """
    Lazily generate permutations where:
      - 0 is always first.
      - Blocks can appear in any order.
      - Elements inside each block can appear in any order.
      - A block remains contiguous.
    """

    # ---------- validation ----------
    expected = set(range(1, n))
    elements = [x for block in blocks for x in block]

    if set(elements) != expected:
        raise ValueError("Blocks must contain every element from 1 to n-1.")

    if len(elements) != len(set(elements)):
        raise ValueError("Blocks contain duplicate elements.")

    if not blocks:
        yield (0,)
        return

    # ---------- recursive generation ----------

    def generate(current, remaining_blocks):
        # All blocks have been consumed
        if not remaining_blocks:
            yield tuple(current)
            return

        # Choose the next block
        for block_index, block in enumerate(remaining_blocks):

            next_remaining = (
                remaining_blocks[:block_index]
                + remaining_blocks[block_index + 1 :]
            )

            # Permute elements inside this block lazily
            for block_perm in permutations(block):

                # Append this block
                current.extend(block_perm)

                yield from generate(current, next_remaining)

                # Backtrack
                del current[-len(block_perm) :]

    yield from generate([0], tuple(blocks))


def generate_block_sequences(block, valid_edges, start=None):
    """
    Lazily generate valid permutations of one block.
    Only expands a permutation if every newly created edge is in valid_edges.
    """

    block = set(block)

    def dfs(path, remaining):
        if not remaining:
            yield tuple(path)
            return

        for node in remaining:

            # Check edge before adding node
            if path:
                edge = (path[-1], node)

                if edge not in valid_edges:
                    continue  # PRUNE immediately

            path.append(node)

            yield from dfs(path, remaining - {node})

            path.pop()

    if start is not None:
        yield from dfs([start], block - {start})
    else:
        for start_node in block:
            yield from dfs([start_node], block - {start_node})


if __name__ == "__main__":
    n = 9
    blocks = ((1, 2, 5), (3, 4, 6), (7, 8))
    p = generate_block_permutations(n, blocks)

    i = 0
    while True:
        try:
            s = next(p)
            print(s)
            i += 1
        except StopIteration:
            break
    print(f"==>> total: {i}")

    block = (1, 2, 5)

    valid_edges = {
        (1, 5),
        (5, 2),
        (2, 1),
        (5, 1),
        (2, 5),
        (1, 2),
    }

    for p in generate_block_sequences(block, valid_edges):
        print(p)

    # n = 9
    # blocks = (1, 2, 5)
    # p = generate_block_permutations_first(n, blocks)
    # i = 0
    # while True:
    #     try:
    #         s = next(p)
    #         i += 1
    #         print(s)
    #     except StopIteration:
    #         break
    # print(f"==>> total: {i}")

    # ntasks = 9
    # nn = {
    #     0: [1, 2, 3, 4, 5, 6, 7, 8],
    #     1: [0, 2, 3, 4, 5, 6, 7, 8],
    #     2: [0, 1, 3, 4, 5, 6, 7, 8],
    #     3: [0, 1, 2, 4, 5, 6, 7, 8],
    #     4: [0, 1, 2, 3, 5, 6, 7, 8],
    #     5: [0, 1, 2, 3, 4, 6, 7, 8],
    #     6: [0, 1, 2, 3, 4, 5, 7, 8],
    #     7: [0, 1, 2, 3, 4, 5, 6, 8],
    #     8: [0, 1, 2, 3, 4, 5, 6, 7],
    # }

    # adjm = _nn_dict_to_adjmat(nn)
    # print(f"==>> adjm: \n{adjm}")

    # path = forwardsearch_hamiltonian_cycle(adjm)
    # if path[0] == -1:
    #     print("Solution does not Exist")
    # else:
    #     print(path)

    # p = generate_brute_force_permutations(300)
    # d = next(p)
    # print(f"==>> d: \n{d}")

    # edges = [
    #     (0, 1),
    #     (0, 2),
    #     (0, 3),
    #     (0, 4),
    #     (0, 5),
    #     (0, 6),
    #     (0, 7),
    #     (0, 8),
    #     (1, 2),
    #     (1, 3),
    #     (1, 4),
    #     (1, 5),
    #     (1, 6),
    #     (1, 7),
    #     (1, 8),
    #     (2, 3),
    #     (2, 4),
    #     (2, 5),
    #     (2, 6),
    #     (2, 7),
    #     (2, 8),
    #     (3, 4),
    #     (3, 5),
    #     (3, 6),
    #     (3, 7),
    #     (3, 8),
    #     (4, 5),
    #     (4, 6),
    #     (4, 7),
    #     (4, 8),
    #     (5, 6),
    #     (5, 7),
    #     (5, 8),
    #     (6, 7),
    #     (6, 8),
    #     (7, 8),
    # ]
    # order = generate_spectral_graph_permutations(edges, n=9, start=0)
    # print(f"==>> order: \n{order}")
