import os
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from problem_planarrr import PlanarRR, RobotScene
from geometric_ellipse import *

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

shapes = {
    # "shape1": {"x": -0.7, "y": 1.3, "h": 2, "w": 2.2},
    "shape1": {"x": -0.7, "y": 2.1, "h": 2, "w": 2.2},
    "shape2": {"x": 2, "y": -2.0, "h": 1, "w": 4.0},
    "shape3": {"x": -3, "y": -3, "h": 1.25, "w": 2},
}
obstacles = [
    box(k["x"], k["y"], k["x"] + k["w"], k["y"] + k["h"]) for k in shapes.values()
]
robot = PlanarRR()
scene = RobotScene(robot, obstacles)


def collision_check(q):
    best, res = scene.distance_to_obstacles(q)
    if best["distance"] <= 0:
        return True
    else:
        return False


dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))

limits = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
num = 100
qrand = np.random.uniform(limits[:, 0], limits[:, 1], size=(num, limits.shape[0]))
print("Random configuration:", qrand)

qrandcheck = np.zeros((num, 1))
for i in range(num):
    q = qrand[i, :].reshape(-1, 1)
    in_collision = collision_check(q)
    if in_collision:
        qrandcheck[i, 0] = 1
    else:
        qrandcheck[i, 0] = 0
print("Collision check results:", qrandcheck.flatten())

qs = np.array([0.15, 0.60]).reshape(-1, 1)
qg = np.array([2.5, 1.5]).reshape(-1, 1)

cmin = np.linalg.norm(qg - qs)
cMaxguess = 1.5 * cmin
Xinf = informed_sampling_bulk(qs, qg, cMaxguess, 1000)
Xinf_surf = informed_surface_sampling_bulk(qs, qg, cMaxguess, 1000)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2], cmap="bwr", s=3)
ax.scatter(qrand[:, 0], qrand[:, 1], c=qrandcheck.flatten(), cmap="gray", s=50)
ax.scatter(Xinf_surf[:, 0], Xinf_surf[:, 1], s=5, c="y", label="informed")
ax.scatter(qs[0], qs[1], c="green", marker="*", s=200, label="start")
ax.scatter(qg[0], qg[1], c="red", marker="*", s=200, label="goal")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("C-space dataset")
ax.set_aspect("equal")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
plt.show()
