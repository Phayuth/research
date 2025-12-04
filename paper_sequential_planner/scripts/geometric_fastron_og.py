import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
trainsize = 600
samples_id = np.random.choice(
    range(dataset.shape[0]), size=trainsize, replace=False
)
dataset_samples = dataset[samples_id]
data = dataset_samples[:, 0:2]
y = dataset_samples[:, 2]


# Fastron
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
    print("Fastron Original Kernel Updating ....")
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

    num_samples = 360
    theta1_samples = np.linspace(-np.pi, np.pi, num_samples)
    theta2_samples = np.linspace(-np.pi, np.pi, num_samples)
    cspace_obs = []

    for i in range(num_samples):
        for j in range(num_samples):
            print(i, j)
            theta = np.array([theta1_samples[i], theta2_samples[j]])
            collision = eval(theta, data, alpha, g)
            if collision == 1:
                cspace_obs.append((theta1_samples[i], theta2_samples[j]))
    cspace_obs = np.array(cspace_obs)

    fig, ax = plt.subplots()
    ax.plot(
        cspace_obs[:, 0],
        cspace_obs[:, 1],
        "ro",
        markersize=2,
        label="Fastron C-space obstacle",
    )
    ax.set_xlabel("Theta 1")
    ax.set_ylabel("Theta 2")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", "box")
    ax.set_title("Fastron C-space Obstacle Approximation")
    ax.legend()
    plt.show()
