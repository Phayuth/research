import math
import time
import matplotlib.pyplot as plt
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
Square = namedtuple(
    "Square", ["lx", "ly"]
)  # lower-left corner (axis-aligned), side = w


# ---------- Helper functions ----------


def points_in_square(points, sq, w):
    lx, ly = sq.lx, sq.ly
    rx, ry = lx + w, ly + w
    return [i for i, p in enumerate(points) if lx <= p.x <= rx and ly <= p.y <= ry]


def generate_candidates(points, w):
    xs = sorted({p.x for p in points})
    ys = sorted({p.y for p in points})
    return [Square(x, y) for x in xs for y in ys]


def greedy_cover(points, w):
    n = len(points)
    uncovered = set(range(n))
    candidates = generate_candidates(points, w)
    cover_sets = [set(points_in_square(points, c, w)) for c in candidates]

    chosen = []
    while uncovered:
        best_idx = max(
            range(len(candidates)), key=lambda i: len(cover_sets[i] & uncovered)
        )
        covered_now = cover_sets[best_idx] & uncovered
        if not covered_now:
            # fallback: if no progress, drop a square at uncovered point
            i = uncovered.pop()
            p = points[i]
            chosen.append(Square(p.x, p.y))
            uncovered -= set(points_in_square(points, chosen[-1], w))
        else:
            chosen.append(candidates[best_idx])
            uncovered -= covered_now
    return chosen


# ---------- Plotting ----------


def plot_cover(points, w, squares, title="Square Cover"):
    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys = [p.x for p in points], [p.y for p in points]
    ax.scatter(xs, ys, c="black", s=30, label="Points", zorder=5)

    # plot each square
    for i, sq in enumerate(squares):
        rect = plt.Rectangle(
            (sq.lx, sq.ly),
            w,
            w,
            edgecolor=f"C{i%10}",
            facecolor=f"C{i%10}",
            alpha=0.2,
            lw=2,
        )
        ax.add_patch(rect)
        ax.text(
            sq.lx + w / 2,
            sq.ly + w / 2,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_aspect("equal")
    ax.set_title(title + f" ({len(squares)} squares)")
    ax.legend()
    plt.show()


# ---------- Example ----------

if __name__ == "__main__":
    import numpy as np

    xy = np.random.uniform(0, 5, (500, 2))
    pts = [Point(x, y) for x, y in xy]
    w = 1.0

    squares_greedy = greedy_cover(pts, w)
    plot_cover(pts, w, squares_greedy, "Greedy Cover")
