import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import rbf_kernel

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
trainsize = 1000
samples_id = np.random.choice(
    range(dataset.shape[0]), size=trainsize, replace=False
)
dataset_samples = dataset[samples_id]
data = dataset_samples[:, 0:2]
y = dataset_samples[:, 2]


# Fastron
N = data.shape[0]  # number of datapoint = number of row the dataset
d = data.shape[1]  # number of dimension = number of col the dataset (x1, ..., xn)
gamma = 10  # kernel width
beta = 100  # conditional bias
maxUpdate = 25  # max update iteration
maxSupportPoints = 1500  # max support points
G = np.zeros((N, N))  # kernel gram matrix guassian kernel of dataset

alpha = np.zeros(N)  # weight, init at zero
# alpha = np.random.uniform(0.0, 1.0, size=(N))  # weight, init at random
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


# def kernel_vector_to_query(data, queryPoint, gamma):
#     """Compute vector of RBF kernel values between each row in data and queryPoint.
#     Returns k of shape (N,)
#     """
#     query = np.asarray(queryPoint)
#     diffs = data - query
#     sq_dists = np.sum(diffs * diffs, axis=1)
#     return np.exp(-gamma * sq_dists)


# def hypothesisf_vectorized(queryPoint, data, alpha, gamma):
#     """Vectorized hypothesis: sign(alpha . k(queryPoint))"""
#     k = kernel_vector_to_query(data, queryPoint, gamma)
#     score = alpha.dot(k)
#     return np.sign(score)


# def eval_vectorized(queryPoint, data, alpha, gamma):
#     """Same as hypothesisf_vectorized, keeps the name `eval` used in original script."""
#     return hypothesisf_vectorized(queryPoint, data, alpha, gamma)


# def F_from_alpha(G, alpha):
#     """Compute F = G.dot(alpha) (raw scores)."""
#     return G.dot(alpha)


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


# G = rbf_kernel(data, data, gamma=gamma)
# alpha, F = original_kernel_update(alpha, F, data, y, G, N, gamma, maxUpdate)


# # save trained model
# np.save(os.path.join(rsrc, "fastron_og_alpha.npy"), alpha)
# np.save(os.path.join(rsrc, "fastron_og_data.npy"), data)
# np.save(os.path.join(rsrc, "fastron_og_labels.npy"), y)
# np.save(os.path.join(rsrc, "fastron_og_gram.npy"), G)
# np.save(os.path.join(rsrc, "fastron_og_F.npy"), F)

# load trained model
alpha = np.load(os.path.join(rsrc, "fastron_og_alpha.npy"))
data = np.load(os.path.join(rsrc, "fastron_og_data.npy"))
y = np.load(os.path.join(rsrc, "fastron_og_labels.npy")).astype(int)
G = np.load(os.path.join(rsrc, "fastron_og_gram.npy"))
F = np.load(os.path.join(rsrc, "fastron_og_F.npy"))

print(f"> alpha: {alpha}")
print(f"> data: {data}")
print(f"> y: {y}")
labelfree = np.where(y == -1)[0]
print(f"> labelfree: {labelfree}")
labelcols = np.where(y == 1)[0]
print(f"> labelcols: {labelcols}")

alpha_nonzero = np.where(alpha != 0.0)[0]
print(f"> alpha_nonzero: {alpha_nonzero}")
print(f"> number of support points: {len(alpha_nonzero)} / {N}")

data_free = data[labelfree]
data_cols = data[labelcols]
data_supp = data[alpha_nonzero]

if __name__ == "__main__":
    queryP = np.array([1, 1])
    collision = eval(queryP, data, alpha, gamma)
    print(f"> collision: {collision}")

    num_samples = 360
    theta1_samples = np.linspace(-np.pi, np.pi, num_samples)
    theta2_samples = np.linspace(-np.pi, np.pi, num_samples)
    cspace_obs = []

    # for i in range(num_samples):
    #     for j in range(num_samples):
    #         print(i, j)
    #         theta = np.array([theta1_samples[i], theta2_samples[j]])
    #         collision = eval(theta, data, alpha, gamma)
    #         if collision == 1:
    #             cspace_obs.append((theta1_samples[i], theta2_samples[j]))
    # cspace_obs = np.array(cspace_obs)
    # np.save(os.path.join(rsrc, "cspace_obstacles_fastron.npy"), cspace_obs)

    cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
    cspace_obs_ft = np.load(os.path.join(rsrc, "cspace_obstacles_fastron.npy"))

    fig, ax = plt.subplots()
    ax.plot(
        cspace_obs[:, 0],
        cspace_obs[:, 1],
        "ro",
        markersize=3,
        label="Geometry C-space obstacle",
    )
    ax.plot(
        cspace_obs_ft[:, 0],
        cspace_obs_ft[:, 1],
        "yo",
        markersize=3,
        label="Fastron C-space obstacle",
        alpha=0.5,
    )
    ax.plot(
        data_free[:, 0],
        data_free[:, 1],
        "go",
        markersize=3,
        label="Fastron dataset free",
        alpha=0.3,
    )
    ax.plot(
        data_cols[:, 0],
        data_cols[:, 1],
        "ko",
        markersize=3,
        label="Fastron dataset obstacle",
        alpha=0.3,
    )
    ax.plot(
        data_supp[:, 0],
        data_supp[:, 1],
        "mx",
        markersize=5,
        label="Fastron support points",
        alpha=0.7,
    )
    ax.set_xlabel("Theta 1")
    ax.set_ylabel("Theta 2")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", "box")
    ax.set_title("Fastron C-space Obstacle Approximation")
    ax.legend()
    plt.show()
