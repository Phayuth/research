import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

np.set_printoptions(linewidth=1000, suppress=True, precision=2)
np.random.seed(42)
rsrc = os.environ["RSRC_DIR"]

# =========================
# space learning
N_TRAIN = 1000
sigma = 0.5
epochs = 200


def svm_train(X_train, y_train, sigma):
    N = X_train.shape[0]
    K = rbf_kernel(X_train, X_train, gamma=1 / (2 * sigma**2))
    alpha = np.zeros(N)

    for _ in range(epochs):
        for i in range(N):
            f_i = np.sum(alpha * y_train * K[:, i])
            if y_train[i] * f_i <= 0:
                alpha[i] += 1
    return alpha


def f_hat(X, X_train, alpha, y_train, sigma):
    Kx = rbf_kernel(X, X_train, gamma=1 / (2 * sigma**2))
    return Kx @ (alpha * y_train)


def sign_correction(xfreeguarantee, f_hat, alpha, X_train, y_train, sigma):
    if f_hat(xfreeguarantee, X_train, alpha, y_train, sigma)[0] < 0:
        alpha *= -1
    return alpha


def save_model(alpha, X_train, y_train):
    # save trained model
    np.save(os.path.join(rsrc, "svmtron_alpha.npy"), alpha)
    np.save(os.path.join(rsrc, "svmtron_X_train.npy"), X_train)
    np.save(os.path.join(rsrc, "svmtron_y_train.npy"), y_train)


def load_model():
    # load trained model
    alpha = np.load(os.path.join(rsrc, "svmtron_alpha.npy"))
    X_train = np.load(os.path.join(rsrc, "svmtron_X_train.npy"))
    y_train = np.load(os.path.join(rsrc, "svmtron_y_train.npy"))
    return alpha, X_train, y_train


# =========================
# edge cost estimation
N_NODES = 200  # number of sampled nodes
tau = 0.5  # confidence threshold
dof = 2  # degrees of freedom
jointlim = np.array([[-np.pi, np.pi]] * dof)  # joint limits


def soft_adjacency(X, f, sigma=0.6):
    W = rbf_kernel(X, X, gamma=1 / (2 * sigma**2))
    conf = 1 / (1 + np.exp(-f))  # sigmoid
    W *= np.minimum(conf[:, None], conf[None, :])
    np.fill_diagonal(W, 0)
    return W


def soft_adjacency_mid_penalty(X, f, sigma=0.6, f_hat=None, f_hat_args=()):
    W = rbf_kernel(X, X, gamma=1 / (2 * sigma**2))
    conf = 1 / (1 + np.exp(-f))  # sigmoid
    W *= np.minimum(conf[:, None], conf[None, :])

    # mid penalty (must optimize later for memory efficiency)
    # centers = (X_nodes[:, None, :] + X_nodes[None, :, :]) / 2
    # midpoints_flat = centers.reshape(-1, 2)
    # values = f_hat(midpoints_flat).reshape(len(X_nodes), len(X_nodes))
    # p = 1 / (1 + np.exp(-values))
    # W *= p
    for i in range(len(X)):  # loop s to e (efficient memory-wise since s-e = e-s)
        for j in range(i + 1, len(X)):
            m = 0.5 * (X[i] + X[j])
            p = 1 / (1 + np.exp(-f_hat(m[None], *f_hat_args)[0]))
            W[i, j] *= p
            W[j, i] *= p
    np.fill_diagonal(W, 0)
    return W


def soft_adjacency_local(X, f, sigma=0.6, k=15):
    N = len(X)
    W = np.zeros((N, N))

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    conf = 1 / (1 + np.exp(-f))
    for i in range(N):
        for j, d in zip(indices[i][1:], distances[i][1:]):
            w = np.exp(-(d**2) / (2 * sigma**2))
            w *= np.minimum(conf[i], conf[j])
            W[i, j] = W[j, i] = w
    return W


def soft_adjacency_radius(X, f, sigma=0.6, r_conn=0.6):
    N = len(X)
    W = np.zeros((N, N))

    nbrs = NearestNeighbors(radius=r_conn).fit(X)
    indices = nbrs.radius_neighbors(X, return_distance=False)
    conf = 1 / (1 + np.exp(-f))
    for i in range(N):
        for j in indices[i]:
            if j == i:
                continue
            d = np.linalg.norm(X[i] - X[j])
            w = np.exp(-(d**2) / (2 * sigma**2))
            w *= min(conf[i], conf[j])
            W[i, j] = w
            W[j, i] = w
    return W


def estimate_adj_matrix(start, goal, X_nodes, f_hat=None, f_hat_args=()):
    # sample nodes in C-space and evaluate f_hat
    f_nodes = f_hat(X_nodes, *f_hat_args)

    # confidence threshold filter
    mask = f_nodes > tau
    X_nodes = X_nodes[mask]
    f_nodes = f_nodes[mask]

    # compute f_hat for start and goal
    X_all = np.vstack([start, goal, X_nodes])
    startf = f_hat(start[None], *f_hat_args)[0:1]
    goalf = f_hat(goal[None], *f_hat_args)[0:1]
    f_all = np.concatenate([startf, goalf, f_nodes])

    # similarity adjacency
    W_all = soft_adjacency_mid_penalty(X_all, f_all, sigma, f_hat, f_hat_args)

    # compute cost adjacency
    C = 1 / (W_all + 1e-6)

    return C, X_all


def estimate_shortest_path(C, X_all):
    dist, pred = shortest_path(C, return_predecessors=True)

    # reconstruct path
    path = []
    j = 1  # goal index
    while j != -9999:
        path.append(j)
        j = pred[0, j]
    path = path[::-1]
    pathq = X_all[path]

    # compute path length in euclidean space
    pathl = 0.0
    for i in range(len(path) - 1):
        pathl += np.linalg.norm(X_all[path[i + 1]] - X_all[path[i]])
    return pathq, pathl


if __name__ == "__main__":
    dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
    samples_id = np.random.choice(
        range(dataset.shape[0]), size=N_TRAIN, replace=False
    )
    dataset_samples = dataset[samples_id]
    X_train = dataset_samples[:, 0:2]
    y = dataset_samples[:, 2]
    y_train = np.where(y <= 0, -1, +1)  # switch sign

    # train SVM
    alpha = svm_train(X_train, y_train, sigma)
    xfreeguarantee = np.array([[0.0, 0.0]])
    alpha = sign_correction(xfreeguarantee, f_hat, alpha, X_train, y_train, sigma)

    # visualize learned f_hat
    grid = 200
    xs = np.linspace(-np.pi, np.pi, grid)
    ys = np.linspace(-np.pi, np.pi, grid)
    XX, YY = np.meshgrid(xs, ys)
    XY = np.column_stack([XX.ravel(), YY.ravel()])
    Z = f_hat(XY, X_train, alpha, y_train, sigma).reshape(grid, grid)
    fig, ax = plt.subplots()
    ax.contourf(XX, YY, Z, levels=50, cmap="coolwarm")
    ax.contour(XX, YY, Z, levels=[0], colors="black")
    ax.set_aspect("equal")
    ax.set_title("Learned free-space scalar field f_hat(x)")
    plt.show()

    # estimate adjacency matrix
    X_nodes = np.random.uniform(
        low=jointlim[:, 0], high=jointlim[:, 1], size=(N_NODES, dof)
    )
    start = np.array([-2.5, -2.5])
    goal = np.array([2.5, 2.5])
    C, X_all = estimate_adj_matrix(
        start, goal, X_nodes, f_hat, (X_train, alpha, y_train, sigma)
    )
    pathq, pathl = estimate_shortest_path(C, X_all)

    # visualize path
    fig, ax = plt.subplots()
    ax.contourf(XX, YY, Z, levels=50, cmap="coolwarm", alpha=0.4)
    ax.scatter(X_all[:, 0], X_all[:, 1], s=5, c="blue")
    ax.plot(pathq[:, 0], pathq[:, 1], "k-", linewidth=2)
    ax.scatter(*start, c="green", s=80)
    ax.scatter(*goal, c="red", s=80)
    ax.set_aspect("equal")
    ax.set_title("Path via learned free-space connectivity (no collision checks)")
    plt.show()
