import math
from itertools import permutations
import tqdm
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def plot_permutation():
    n = 100
    numes = np.linspace(0, n, num=n, dtype=int)
    lenl = scipy.special.factorial(numes)

    fig, ax = plt.subplots()
    ax.plot(numes, lenl)
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Number of permutations (n!)")
    ax.set_title("Number of permutations for n elements in log scale")
    ax.set_yscale("log")
    ax.set_xlim([0, n])
    plt.show()


def brute_force_permutations():
    numelement = 9
    items = range(0, numelement)
    perms = permutations(items)
    lenl = math.factorial(numelement)

    best_perm = None
    cost = np.inf
    pbar = tqdm.tqdm(perms, total=lenl, desc="Processing permutations")
    for p in pbar:
        c = 0
        for i in range(len(p) - 1):
            c += abs(p[i] - p[i + 1])
        if c < cost:
            cost = c
            best_perm = p
        pbar.set_postfix({"perm": p, "cost": c, "best_cost": cost})

    print(f"==>> best_perm: \n{best_perm}")
    print(f"==>> cost: \n{cost}")


def first_0_fixed_element():
    # if we fixed the first element to be 0, we would have 8! permutations
    # it not 9! but 9!/9
    numelement = 9
    items = range(numelement)
    perms = [(0,) + p for p in permutations(items[1:])]

    best_perm = None
    cost = np.inf
    pbar = tqdm.tqdm(perms, desc="Processing permutations")
    for p in pbar:
        c = 0
        for i in range(len(p) - 1):
            c += abs(p[i] - p[i + 1])
        if c < cost:
            cost = c
            best_perm = p
        pbar.set_postfix({"perm": p, "cost": c, "best_cost": cost})

    print(f"==>> best_perm: \n{best_perm}")
    print(f"==>> cost: \n{cost}")


def consecutive_first_block():
    others = [3, 4, 5, 6, 7, 8]
    perms = []
    for block in [(0, 1, 2), (0, 2, 1)]:
        for p in permutations(others):
            perms.append(block + p)

    print(len(perms))  # 1440


def perm_gen_hardblock():

    def generate_permutations(n, blocks):
        blocks = [set(b) for b in blocks]

        # elements that participate in each block
        element_blocks = defaultdict(list)
        for b in blocks:
            for x in b:
                element_blocks[x].append(b)

        results = []

        def check_partial(path, remaining):
            pos = {v: i for i, v in enumerate(path)}

            for block in blocks:
                placed = block.intersection(pos.keys())

                # If part of a block is placed, the whole block must eventually
                # fit inside a consecutive interval.
                if len(placed) > 1:
                    positions = [pos[x] for x in placed]

                    span = max(positions) - min(positions) + 1

                    # Number of empty positions available inside the span
                    missing_inside = span - len(placed)

                    # If span is already larger than block size, impossible
                    if span > len(block):
                        return False

            return True

        def backtrack(path, unused):
            if len(path) == n:
                results.append(tuple(path))
                return

            for x in list(unused):

                new_path = path + [x]
                new_unused = unused - {x}

                if check_partial(new_path, new_unused):
                    backtrack(new_path, new_unused)

        # Fix 0 as first element (cyclic uniqueness)
        backtrack([0], set(range(1, n)))

        return results

    # Example
    n = 9

    blocks = [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
        (5, 6, 7),
        (6, 7, 8),
    ]

    valid = generate_permutations(n, blocks)

    print(len(valid))
    for p in valid[:10]:
        print(p)


def perm_gen_spectral_ordering(edges, n, start=0):
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

    # Build symmetric adjacency matrix
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
    order = list(np.argsort(fiedler))

    # Rotate so that `start` is first
    idx = order.index(start)
    order = order[idx:] + order[:idx]

    return order


def perm_gen_backtracking(n, edges, max_distance):
    """
    Generate ALL permutations satisfying:
        abs(pos[i] - pos[j]) <= max_distance
    for every edge (i, j).

    0 is fixed at the first position to remove
    cyclic-rotation duplicates.
    """

    # Build adjacency
    neighbors = [[] for _ in range(n)]

    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Position of each element.
    # -1 means not placed yet.
    pos = [-1] * n

    # 0 is always first
    pos[0] = 0

    path = [0]

    def can_place(x, position):
        """
        Check whether x can be placed at `position`
        without violating already-placed neighbors.
        """

        for y in neighbors[x]:
            if pos[y] != -1:
                if abs(position - pos[y]) > max_distance:
                    return False

        return True

    def backtrack():
        if len(path) == n:
            yield tuple(path)
            return

        position = len(path)

        for x in range(1, n):

            if pos[x] != -1:
                continue

            if not can_place(x, position):
                continue

            # Place
            pos[x] = position
            path.append(x)

            yield from backtrack()

            # Undo
            path.pop()
            pos[x] = -1

    yield from backtrack()


if __name__ == "__main__":
    # plot_permutation()
    # brute_force_permutations()
    # first_0_fixed_element()
    # consecutive_first_block()

    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
        (5, 6),
        (5, 7),
        (5, 8),
        (6, 7),
        (6, 8),
        (7, 8),
    ]
    # order = perm_gen_spectral_ordering(edges, n=9, start=0)
    # print(f"==>> order: \n{order}")

    n = 9
    valid_perms = perm_gen_backtracking(n, edges, max_distance=1)
    for p in valid_perms:
        print(p)
