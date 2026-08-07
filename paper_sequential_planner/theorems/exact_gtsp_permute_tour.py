import math
from itertools import permutations
import tqdm
import numpy as np

numelement = 9


def brute_force_permutations():
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

if __name__ == "__main__":
    # brute_force_permutations()
    # first_0_fixed_element()
    consecutive_first_block()