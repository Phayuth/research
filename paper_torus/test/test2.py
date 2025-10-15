import math
from collections import namedtuple
import matplotlib.pyplot as plt
import itertools
import time

Rect = namedtuple("Rect", ["cx", "cy", "w", "h"])
Square = namedtuple("Square", ["lx", "ly"])  # lower-left corner


def rect_edges(rect):
    cx, cy, rw, rh = rect
    left = cx - rw / 2.0
    right = cx + rw / 2.0
    bottom = cy - rh / 2.0
    top = cy + rh / 2.0
    return left, right, bottom, top


def feasible_interval_for_square(rect, square_w):
    left, right, bottom, top = rect_edges(rect)
    # feasible lx range and ly range for lower-left corner so square fully contains rect
    lx_min = right - square_w
    lx_max = left
    ly_min = top - square_w
    ly_max = bottom
    # interval might be reversed if empty; return None when impossible
    if lx_min > lx_max or ly_min > ly_max:
        return None
    return (lx_min, lx_max), (ly_min, ly_max)


def generate_candidates_from_endpoints(rects, w):
    # For each rectangle we have two x-endpoints and two y-endpoints for feasible ranges:
    # endpoints: (right-w) and (left) for x; (top-w) and (bottom) for y.
    xs = set()
    ys = set()
    feasible = []
    for r in rects:
        intervals = feasible_interval_for_square(r, w)
        if intervals is None:
            feasible.append(None)
            continue
        (lx_min, lx_max), (ly_min, ly_max) = intervals
        xs.add(lx_min)
        xs.add(lx_max)
        ys.add(ly_min)
        ys.add(ly_max)
        feasible.append(((lx_min, lx_max), (ly_min, ly_max)))
    # create candidate grid from all endpoint combinations
    candidates = [
        Square(x, y) for x, y in itertools.product(sorted(xs), sorted(ys))
    ]
    return candidates, feasible


def covers_square(square, rect, w):
    # check if square at lower-left 'square' with side w fully contains rectangle 'rect'
    intervals = feasible_interval_for_square(rect, w)
    if intervals is None:
        return False
    (lx_min, lx_max), (ly_min, ly_max) = intervals
    return (lx_min - 1e-12) <= square.lx <= (lx_max + 1e-12) and (
        ly_min - 1e-12
    ) <= square.ly <= (ly_max + 1e-12)


def build_cover_sets(candidates, rects, w):
    cover_sets = []
    for c in candidates:
        covered = set(i for i, r in enumerate(rects) if covers_square(c, r, w))
        cover_sets.append(covered)
    return cover_sets


# Greedy set-cover (approx)
def greedy_cover_rectangles(rects, w):
    # feasibility check
    for i, r in enumerate(rects):
        if feasible_interval_for_square(r, w) is None:
            raise ValueError(
                f"Rectangle {i} (w={r.w}, h={r.h}) cannot be covered by square size {w}."
            )
    candidates, feasible = generate_candidates_from_endpoints(rects, w)
    cover_sets = build_cover_sets(candidates, rects, w)
    uncovered = set(range(len(rects)))
    chosen = []
    while uncovered:
        best_idx = max(
            range(len(candidates)), key=lambda i: len(cover_sets[i] & uncovered)
        )
        new = cover_sets[best_idx] & uncovered
        if not new:
            # fallback: pick a candidate that covers at least one uncovered (shouldn't happen normally)
            # but to be safe, pick a square anchored at that rectangle's feasible lower-left min
            rid = next(iter(uncovered))
            (lx_min, lx_max), (ly_min, ly_max) = feasible[rid]
            c = Square(lx_min, ly_min)
            chosen.append(c)
            uncovered -= set(
                i for i, r in enumerate(rects) if covers_square(c, r, w)
            )
        else:
            chosen.append(candidates[best_idx])
            uncovered -= new
    return chosen


# Small exact branch-and-bound (search over candidate set)
def exact_cover_rectangles(rects, w, time_limit=5.0):
    start = time.time()
    # feasibility check
    for i, r in enumerate(rects):
        if feasible_interval_for_square(r, w) is None:
            raise ValueError(
                f"Rectangle {i} cannot be covered by square size {w}."
            )
    candidates, _ = generate_candidates_from_endpoints(rects, w)
    cover_sets = build_cover_sets(candidates, rects, w)
    # remove empty candidate covers
    filtered = [(c, s) for c, s in zip(candidates, cover_sets) if s]
    candidates = [c for c, s in filtered]
    cover_sets = [s for c, s in filtered]
    n = len(rects)
    allset = set(range(n))
    # simple greedy for initial upper bound
    try:
        greedy = greedy_cover_rectangles(rects, w)
        best_k = len(greedy)
        best_sol = greedy
    except ValueError:
        raise
    # order by size desc to help pruning
    order = sorted(range(len(cover_sets)), key=lambda i: -len(cover_sets[i]))
    cover_sets = [cover_sets[i] for i in order]
    candidates = [candidates[i] for i in order]
    max_cov = max((len(s) for s in cover_sets), default=0)

    def dfs(idx, chosen, covered):
        nonlocal best_k, best_sol
        if time.time() - start > time_limit:
            return
        if len(chosen) >= best_k:
            return
        if covered == allset:
            best_k = len(chosen)
            best_sol = list(chosen)
            return
        remaining = len(allset - covered)
        if max_cov == 0:
            return
        lower_more = math.ceil(remaining / max_cov)
        if len(chosen) + lower_more >= best_k:
            return
        for i in range(idx, len(cover_sets)):
            new = cover_sets[i] - covered
            if not new:
                continue
            chosen.append(candidates[i])
            dfs(i + 1, chosen, covered | cover_sets[i])
            chosen.pop()
            if time.time() - start > time_limit:
                return

    dfs(0, [], set())
    return best_sol


# Plotting rectangles and chosen squares
def plot_solution(rects, w, squares, title="Cover"):
    fig, ax = plt.subplots(figsize=(7, 7))
    # plot rectangles (input)
    for i, r in enumerate(rects):
        left, right, bottom, top = rect_edges(r)
        rect_patch = plt.Rectangle(
            (left, bottom), r.w, r.h, edgecolor="black", facecolor="none", lw=1.5
        )
        ax.add_patch(rect_patch)
        ax.text(
            (left + right) / 2,
            (bottom + top) / 2,
            str(i),
            ha="center",
            va="center",
        )
    # plot chosen squares
    for i, sq in enumerate(squares):
        ax.add_patch(
            plt.Rectangle(
                (sq.lx, sq.ly),
                w,
                w,
                edgecolor=f"C{i%10}",
                facecolor=f"C{i%10}",
                alpha=0.25,
                lw=2,
            )
        )
        ax.text(sq.lx + w / 2, sq.ly + w / 2, f"S{i}", ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_title(f"{title} ({len(squares)} squares)")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.show()


# Example usage
if __name__ == "__main__":
    # create rectangles: Rect(cx,cy,width,height)
    import numpy as np
    xy = np.random.uniform(-5, 5, (300, 2))
    wh = np.random.uniform(0.5, 0.5, (300, 2))
    rects = [Rect(x, y, w, h) for (x, y), (w, h) in zip(xy, wh)]
    # rects = [

    #     Rect(0.2, 0.2, 0.2, 0.2),
    #     Rect(0.4, 0.25, 0.3, 0.25),
    #     Rect(1.1, 0.9, 0.5, 0.4),
    #     Rect(2.0, 2.1, 0.4, 0.3),
    #     Rect(3.0, 2.9, 0.6, 0.5),
    # ]
    w = 0.8

    # greedy solution
    greedy_sol = greedy_cover_rectangles(rects, w)
    print("Greedy chose", len(greedy_sol), "squares:", greedy_sol)
    plot_solution(rects, w, greedy_sol, "Greedy Cover")

    # try small exact (may take time for many rects)
    try:
        exact_sol = exact_cover_rectangles(rects, w, time_limit=3.0)
        print("Exact best found", len(exact_sol), "squares:", exact_sol)
        plot_solution(rects, w, exact_sol, "Exact (best found)")
    except ValueError as e:
        print("Feasibility error:", e)
