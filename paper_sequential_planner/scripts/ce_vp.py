# 2D Via-Point Sampling Example (No CMA-ES, Just Correlated Sampling)
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Start and goal
q_start = np.array([0.0, 0.0])
q_goal = np.array([10.0, 10.0])

# Initial mean for 2 via-points in 2D (q1x, q1y, q2x, q2y)
mean = np.array([3.0, 3.0, 7.0, 7.0])

# Correlated covariance matrix (manually defined)
cov = np.array(
    [
        [1.0, 0.8, 0.5, 0.4],
        [0.8, 1.0, 0.4, 0.5],
        [0.5, 0.4, 1.2, 0.9],
        [0.4, 0.5, 0.9, 1.2],
    ]
)

# Sample 5 candidate via-point sets
num_candidates = 5
samples = np.random.multivariate_normal(mean, cov, size=num_candidates)
print(samples)

# Time discretization
t = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.plot(q_start[0], q_start[1], "go", label="Start")
ax.plot(q_goal[0], q_goal[1], "ro", label="Goal")
ax.plot(
    [q_start[0], mean[0], mean[2], q_goal[0]],
    [q_start[1], mean[1], mean[3], q_goal[1]],
    "k--o",
    label="Mean Path",
)
for s in samples:
    v1 = s[0:2]
    v2 = s[2:4]

    # Cubic Bezier (simple smooth interpolation)
    traj = (
        ((1 - t) ** 3)[:, None] * q_start
        + (3 * (1 - t) ** 2 * t)[:, None] * v1
        + (3 * (1 - t) * t**2)[:, None] * v2
        + (t**3)[:, None] * q_goal
    )

    ax.plot(traj[:, 0], traj[:, 1])
    ax.scatter([v1[0], v2[0]], [v1[1], v2[1]])

ax.scatter(q_start[0], q_start[1])
ax.scatter(q_goal[0], q_goal[1])
ax.set_title("2D Via-Point Sampling with Correlated Gaussian")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
plt.show()
