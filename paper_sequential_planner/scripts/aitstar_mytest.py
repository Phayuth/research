import os
import numpy as np
from problem_planarrr import PlanarRR, RobotScene
from shapely.geometry import LineString, box
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))

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
fig, ax = plt.subplots(figsize=(6, 6))


ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3, alpha=0.1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("C-space obstacles")
ax.set_aspect("equal")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
plt.show()
