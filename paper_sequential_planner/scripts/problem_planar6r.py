import os
import numpy as np
from shapely.geometry import LineString, box, Polygon, MultiPolygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def forward_kinematics_vectorized(joint_angles):
    link_lengths = np.ones(6)  # Assuming all link lengths are 1 unit
    n, num_joints = joint_angles.shape
    cumulative_angles = np.cumsum(joint_angles, axis=1)
    x_displacements = link_lengths * np.cos(cumulative_angles)
    y_displacements = link_lengths * np.sin(cumulative_angles)
    displacements = np.stack((x_displacements, y_displacements), axis=-1)
    positions = np.cumsum(displacements, axis=1)
    base = np.zeros((n, 1, 2))  # Shape (n, 1, 2), to represent the base at (0, 0)
    result = np.concatenate([base, positions], axis=1)  # Shape (n, 7, 2)
    return result


o1 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])
o2 = Polygon([(-4, 3), (-3, 3), (-3, 4), (-4, 4)])
oo = MultiPolygon([o1, o2])
clearence = 0.1
rb = 0.3


def get_dmin(p):
    dmin = LineString(p).buffer(rb).distance(oo)
    return dmin


# my parallel version is not working well yet
def compute_obstacle_cost_one_traj(traj):
    xy_coordinates = forward_kinematics_vectorized(traj)
    armcols = [LineString(pp).buffer(rb) for pp in xy_coordinates]
    dmin = np.array([armcol.distance(oo) for armcol in armcols])
    cost = np.maximum(clearence + rb - dmin, 0)
    return np.sum(cost)
