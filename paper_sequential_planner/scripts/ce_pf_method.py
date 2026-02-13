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


def dist_to_obs(q):
    theta = q.reshape((-1, 1))
    best, results = scene.distance_to_obstacles(theta)
    if best is not None:
        return best["distance"]


# ---------- potentials ----------
def U_att(q, qg, k_att=1.0):
    return 0.5 * k_att * np.linalg.norm(q - qg) ** 2


def U_rep(q, d0=0.01, k_rep=1.0, eps=1e-6):
    d = dist_to_obs(q)
    d = max(d, eps)  # prevent zero

    if d >= d0:
        return 0.0

    return 0.5 * k_rep * (1 / d - 1 / d0) ** 2


def U_total(q, qg):
    return U_att(q, qg) + U_rep(q)


# ---------- numerical gradient ----------
def grad_U(q, qg, eps=1e-5):
    g = np.zeros(2)
    for i in range(2):
        dq = np.zeros(2)
        dq[i] = eps
        g[i] = (U_total(q + dq, qg) - U_total(q - dq, qg)) / (2 * eps)
    return g


# ---------- planner ----------
def potential_field_plan(q_start, q_goal, alpha=0.01, max_iter=2000, tol=1e-3):

    q = q_start.copy()
    path = [q.copy()]

    for _ in range(max_iter):
        g = grad_U(q, q_goal)
        q = q - alpha * g
        path.append(q.copy())

        if np.linalg.norm(q - q_goal) < tol:
            break

    return np.array(path)


def potential_field_plan_stochastic(
    q_start, q_goal, alpha=0.05, sigma=0.01, max_iter=2000, tol=1e-3
):

    q = q_start.copy()
    path = [q.copy()]

    for _ in range(max_iter):
        g = grad_U(q, q_goal)
        noise = sigma * np.random.randn(2)  # stochastic term
        q = q - alpha * g + noise
        path.append(q.copy())

        if np.linalg.norm(q - q_goal) < tol:
            break

    return np.array(path)


# ---------- example ----------
if __name__ == "__main__":
    q0 = np.array([0.0, 0.0])
    # qg = np.array([2.0, -3.0])
    qg = np.array([3.0, 3.0])

    traj = potential_field_plan_stochastic(q0, qg)
    print(traj.shape)

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    ax.plot(traj[:, 0], traj[:, 1], "b--", linewidth=2, label="planned path")
    ax.plot(q0[0], q0[1], "ro", markersize=8, label="start")
    ax.plot(qg[0], qg[1], "go", markersize=8, label="goal")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.set_title("Potential Field Planning in C-space")
    plt.show()
