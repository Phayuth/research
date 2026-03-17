import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import alphashape
import shapely

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

dof = 2
n_points = 200
alpha = 4.0
points = np.random.uniform(-np.pi, np.pi, size=(n_points, dof))

gen = alphashape.alphasimplices(points)
tris = []
for simplex, r in gen:
    if r < 1 / alpha:
        tris.append(points[simplex])
print(f"Number of triangles: {len(tris)}")

fig, ax = plt.subplots()
for tri in tris:
    ax.text(
        tri[:, 0].mean(),
        tri[:, 1].mean(),
        "O",
        ha="center",
        va="center",
        fontsize=4,
        color="red",
    )
    ax.add_patch(Polygon(tri, edgecolor="red", facecolor="none"))
ax.set_aspect("equal")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
plt.show()

def collision_check(q):
    if np.random.rand() < 0.1:
        return True
    else:
        return False