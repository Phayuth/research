import numpy as np
import matplotlib.pyplot as plt


class NLinkArm(object):

    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + self.link_lengths[
                i - 1
            ] * np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + self.link_lengths[
                i - 1
            ] * np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T


def detect_collision(line_seg, circle):
    a_vec = np.array([line_seg[0][0], line_seg[0][1]])
    b_vec = np.array([line_seg[1][0], line_seg[1][1]])
    c_vec = np.array([circle[0], circle[1]])
    radius = circle[2]
    line_vec = b_vec - a_vec
    line_mag = np.linalg.norm(line_vec)
    circle_vec = c_vec - a_vec
    proj = circle_vec.dot(line_vec / line_mag)
    if proj <= 0:
        closest_point = a_vec
    elif proj >= line_mag:
        closest_point = b_vec
    else:
        closest_point = a_vec + line_vec * proj / line_mag
    if np.linalg.norm(closest_point - c_vec) > radius:
        return False

    return True


def get_occupancy_grid(arm, obstacles):

    grid = [[0 for _ in range(M)] for _ in range(M)]
    theta_list = [2 * i * np.pi / M for i in range(-M // 2, M // 2 + 1)]

    dataset = []

    for i in range(M):
        for j in range(M):
            arm.update_joints([theta_list[i], theta_list[j]])
            points = arm.points
            collision_detected = False
            for k in range(len(points) - 1):
                for obstacle in obstacles:
                    line_seg = [points[k], points[k + 1]]
                    collision_detected = detect_collision(line_seg, obstacle)
                    if collision_detected:
                        break
                if collision_detected:
                    break
            grid[i][j] = int(collision_detected)

            if int(collision_detected) == 1:
                collision_stat = 1
            elif int(collision_detected) == 0:
                collision_stat = -1
            dataset.append([theta_list[i], theta_list[j], collision_stat])

    return np.array(grid), dataset


# Simulation parameters
M = 25  # number of sample to divide into and number of grid cell
obstacles = [
    [1.75, 0.75, 0.6],
    [0.55, 1.5, 0.5],
    [0, -1, 0.25],
    [-1.5, -1.5, 0.25],
]  # x y radius
arm = NLinkArm([1, 1], [0, 0])
# grid, dataset = get_occupancy_grid(arm, obstacles)

import os

rsrc = os.environ["RSRC_DIR"]
dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
data = dataset[:, 0:2]
y = dataset[:, 2]
trainsize = 600
data = np.random.choice(data, size=trainsize, replace=True)
y = np.random.choice(y, size=trainsize, replace=True)


# Fastron
# dataset = np.array(dataset)
# data = dataset[:, 0:2]
# y = dataset[:, 2]
# print(data.shape)
# print(y.shape)

N = data.shape[0]  # number of datapoint = number of row the dataset has
d = data.shape[
    1
]  # number of dimensionality = number of columns the dataset has (x1, x2, ..., xn)
g = 10  # kernel width
beta = 100  # conditional bias
maxUpdate = 25  # max update iteration
maxSupportPoints = 1500  # max support points
G = np.zeros((N, N))  # kernel gram matrix guassian kernel of dataset

# alpha = np.zeros(N)  # weight, init at zero
alpha = np.random.uniform(0.0, 1.0, size=(N))  # weight, init at random
F = np.zeros(N)  # hypothesis

# active learning parameters
allowance = 800  # number of new samples
kNS = 4  # number of points near supports
sigma = 0.5  # Gaussian sampling std
exploitP = 0.5  # proportion of exploitation samples
gramComputed = np.zeros(N)


def gaussian_kernel(x, y, gamma):
    """Gaussian Kernel measuring similarity between 2 vectors"""
    distance = np.linalg.norm(x - y)
    return np.exp(-gamma * distance**2)


def hypothesisf(queryPoint, data, alpha, gamma):
    term = []
    for i, xi in enumerate(data):
        term.append(alpha[i] * gaussian_kernel(xi, queryPoint, gamma))
    ypred = np.sign(sum(term))
    return ypred


def eval(queryPoint, data, alpha, gamma):
    term = []
    for i, alphai in enumerate(alpha):
        if alphai != 0.0:
            term.append(alphai * gaussian_kernel(data[i], queryPoint, gamma))
    ypred = np.sign(sum(term))
    return ypred


def compute_kernel_gram_matrix(G, data, gamma):
    for i in range(N):
        for j in range(N):
            G[i, j] = gaussian_kernel(data[i], data[j], gamma)
    return G


def original_kernel_update(alpha, F, data, y, G, N, g, maxUpdate):
    """Brute force update, => unneccessary calculation"""
    for iter in range(maxUpdate):
        print(iter)

        # shuffle
        # np.random.shuffle(dataset)
        # data = dataset[:, 0:2]
        # y = dataset[:, 2]

        for i in range(N):
            margini = y[i] * hypothesisf(data[i], data, alpha, g)
            # print(f"> margini: {margini}")
            if margini <= 0:
                alpha[i] += y[i]
                F += y[i] * G[:, i]

    return alpha, F


def onestep_correction_update(alpha, F, data, y, G, N, g, maxUpdate):
    for iter in range(maxUpdate):
        # for iter in range(1):
        print(iter)
        F = np.array([hypothesisf(data[i], data, alpha, g) for i in range(N)])
        marg = y * (F - alpha)
        margcond = marg > 0.0
        alphacond = alpha != 0.0
        condd = margcond & alphacond
        while np.any(condd):
            indices = np.where(condd)[0]
            mn = marg[indices]
            kk = np.argmax(mn)
            j = indices[kk]
            for i in indices:
                F[i] = F[i] - G[i, j] * alpha[j]
            alpha[j] = 0

            marg = y * (F - alpha)
            margcond = marg > 0.0
            alphacond = alpha != 0.0
            condd = margcond & alphacond

        yF = y * F
        if np.all(yF > 0):
            # return alpha, F
            return
        else:
            j = np.argmin(yF)

        if y[j] > 0:
            deltaalpha = beta * y[j] - F[j]
        else:
            deltaalpha = y[j] - F[j]

        alpha[j] += deltaalpha
        F += G[:, j] * deltaalpha

        # for i in range(N):
        #     F[i] = F[i] + G[i, j] * deltaalpha

    return alpha, F


G = compute_kernel_gram_matrix(G, data, g)
alpha, F = original_kernel_update(alpha, F, data, y, G, N, g, maxUpdate)
# alpha, F = onestep_correction_update(alpha, F, data, y, G, N, g, maxUpdate)


if __name__ == "__main__":
    queryP = np.array([1, 1])
    collision = eval(queryP, data, alpha, g)
    print(f"> collision: {collision}")
