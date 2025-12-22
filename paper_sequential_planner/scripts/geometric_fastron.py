import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import rbf_kernel

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


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


def compute_kernel_gram_matrix(data, gamma):
    G = np.zeros((data.shape[0], data.shape[0]))
    for index in np.ndindex(G.shape):
        G[index] = gaussian_kernel(data[index[0]], data[index[1]], gamma)
    return G


def original_kernel_update(alpha, F, data, y, G, N, g, maxUpdate):
    """Brute force update, => unneccessary calculation"""
    print("Fastron Original Kernel Updating ....")
    for iter in range(maxUpdate):
        print(f"Training iteration: {iter}")

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


def fastron_update(alpha, F, data, y, G, N, g, maxUpdate, beta):
    for iter in range(maxUpdate):
        print(f"Training iteration: {iter}")

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


def fastron_active_learning():
    pass


def save_model(alpha, data, y, G, F):
    # save trained model
    np.save(os.path.join(rsrc, "fastron_og_alpha.npy"), alpha)
    np.save(os.path.join(rsrc, "fastron_og_data.npy"), data)
    np.save(os.path.join(rsrc, "fastron_og_labels.npy"), y)
    np.save(os.path.join(rsrc, "fastron_og_gram.npy"), G)
    np.save(os.path.join(rsrc, "fastron_og_F.npy"), F)


def load_model():
    # load trained model
    alpha = np.load(os.path.join(rsrc, "fastron_og_alpha.npy"))
    data = np.load(os.path.join(rsrc, "fastron_og_data.npy"))
    y = np.load(os.path.join(rsrc, "fastron_og_labels.npy")).astype(int)
    G = np.load(os.path.join(rsrc, "fastron_og_gram.npy"))
    F = np.load(os.path.join(rsrc, "fastron_og_F.npy"))
    return alpha, data, y, G, F


class ModelConfig:
    gamma = 10  # kernel width
    beta = 100  # conditional bias
    maxUpdate = 25  # max update iteration
    maxSupportPoints = 1500  # max support points

    # active learning parameters
    allowance = 800  # number of new samples
    kNS = 4  # number of points near supports
    sigma = 0.5  # Gaussian sampling std
    exploitP = 0.5  # proportion of exploitation samples


def train_original_kernel_perceptron_model(data, y, configcls):
    # precompute
    N = data.shape[0]  # num datapoint = number of row the dataset
    d = data.shape[1]  # num dof = number of col the dataset (x1, ..., xn)
    alpha = np.zeros(N)  # weight, init at zero
    F = np.zeros(N)  # hypothesis
    G = rbf_kernel(data, data, gamma=configcls.gamma)  # grammatrix guassian kernel

    # training
    alpha, F = original_kernel_update(
        alpha,
        F,
        data,
        y,
        G,
        N,
        configcls.gamma,
        configcls.maxUpdate,
    )
    return alpha, data, y, G, F


def train_fastron_model(data, y, configcls):
    # precompute Fastron
    N = data.shape[0]
    d = data.shape[1]
    alpha = np.zeros(N)
    F = np.zeros(N)
    G = rbf_kernel(data, data, gamma=configcls.gamma)

    # training
    alpha, F = fastron_update(
        alpha,
        F,
        data,
        y,
        G,
        N,
        configcls.gamma,
        configcls.maxUpdate,
        configcls.beta,
    )
    return alpha, data, y, G, F


if __name__ == "__main__":
    # load or generate data
    data = np.random.uniform(-np.pi, np.pi, (1000, 2))
    y = np.random.choice([-1, 1], size=(1000,))

    # train model
    alpha, data, y, G, F = train_original_kernel_perceptron_model(
        data, y, ModelConfig
    )

    # use model to predict
    queryP = np.array([1, 1])
    collision = eval(queryP, data, alpha, ModelConfig.gamma)
    print(f"Query Point: {queryP}, Collision: {collision}")
