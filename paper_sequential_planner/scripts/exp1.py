# follow the taskspace tour via numerical inverse kinematics
import numpy as np
from spatialmath import SE3


def interpolate_SE3(s1, s2, steps):
    return [s1.interp(s2, t) for t in np.linspace(0, 1, steps)]


s1 = SE3(0, 0, 0)
s2 = SE3(1, 1, 1)
steps = 5
traj = interpolate_SE3(s1, s2, steps)
for i, t in enumerate(traj):
    print(f"step {i}: {t}")
