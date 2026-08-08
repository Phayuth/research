from itertools import permutations


def generate_permutations(n, edges):
    """
    Generate all unique cyclic permutations with 0 fixed first.

    Returns:
        generator of permutations
    """
    for p in permutations(range(1, n)):
        yield (0,) + p


def ordering_cost(perm, edges):
    """
    Sum of positional distances of all KNN edges.

    Smaller = KNN neighbors are more tightly grouped.
    """
    pos = [0] * len(perm)

    for i, node in enumerate(perm):
        pos[node] = i

    cost = 0

    for i, j in edges:
        cost += abs(pos[i] - pos[j])

    return cost


n = 9
edges = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 5),
    (4, 6),
    (4, 7),
    (5, 6),
    (5, 7),
    (5, 8),
    (6, 7),
    (6, 8),
    (7, 8),
]

best_perm = None
best_cost = float("inf")

for perm in generate_permutations(n, edges):
    cost = ordering_cost(perm, edges)
    if cost < best_cost:
        best_cost = cost
        best_perm = perm
print("Best permutation:", best_perm)
print("KNN ordering cost:", best_cost)


candidates = []
for perm in generate_permutations(n, edges):
    cost = ordering_cost(perm, edges)
    candidates.append((cost, perm))
candidates.sort()
for cost, perm in candidates:
    print(cost, perm)

len_candidates = len(candidates)
print(f"==>> len_candidates: \n{len_candidates}")