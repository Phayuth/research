import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import rbf_kernel

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def f_hat(x_query, X_train, alpha, y_train, gamma):
    Kx = rbf_kernel(x_query, X_train, gamma=gamma)
    return Kx @ (alpha * y_train)


def re_corrected_sign(xfreegarantee, X_train, alpha, y_train, gamma):
    if f_hat(xfreegarantee, X_train, alpha, y_train, gamma)[0] < 0:
        alpha *= -1
    return alpha


class ModelConfig:
    sigma = 0.5
    gamma = 1 / (2 * sigma**2)
    epochs = 100


def train_svmtron(X_train, y_train, xfreegarantee, configcls):
    # precompute
    K = rbf_kernel(X_train, X_train, gamma=configcls.gamma)
    alpha = np.zeros(X_train.shape[0])

    for ep in range(configcls.epochs):
        print(f"Epoch {ep+1}/{configcls.epochs}")
        for i in range(X_train.shape[0]):
            f_i = np.sum(alpha * y_train * K[:, i])
            if y_train[i] * f_i <= 0:
                alpha[i] += 1

    alpha = re_corrected_sign(
        xfreegarantee,
        X_train,
        alpha,
        y_train,
        configcls.gamma,
    )
    return alpha, X_train, y_train


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


if __name__ == "__main__":
    # load dataset
    dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
    N_train = 1000
    sample_id = np.random.choice(
        range(dataset.shape[0]), size=(N_train,), replace=False
    )
    dataset_samples = dataset[sample_id]
    data = dataset_samples[:, 0:2]
    y = dataset_samples[:, 2]
    ynew = np.where(y <= 0, -1, 1)  # switch sign
    X_train = data
    y_train = ynew
    xfreegarantee = np.array([0.0, 0.0]).reshape(1, -1)

    # train SVM-Tron model
    alpha, X_train, y_train = train_svmtron(
        X_train, y_train, xfreegarantee, ModelConfig
    )

    # use model to predict
    f = np.array([0.0, 0.0]).reshape(1, -1)
    f_value = f_hat(f, X_train, alpha, y_train, ModelConfig.gamma)
    print(f"f({f[0]}) = {f_value[0]:.4f}")

    grid = 200
    xs = np.linspace(-np.pi, np.pi, grid)
    ys = np.linspace(-np.pi, np.pi, grid)
    XX, YY = np.meshgrid(xs, ys)
    XY = np.column_stack([XX.ravel(), YY.ravel()])
    Z = f_hat(XY, X_train, alpha, y_train, ModelConfig.gamma).reshape(grid, grid)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(XX, YY, Z, levels=50, cmap="coolwarm")
    ax.contour(XX, YY, Z, levels=[0], colors="k", linewidths=2)
    ax.set_aspect("equal")
    ax.set_title("Learned Decision Boundary with SVM-Tron")
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, Z, cmap="coolwarm", alpha=0.8)
    ax.plot_surface(XX, YY, np.zeros_like(Z), color="gray", alpha=0.2)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(np.min(Z), np.max(Z))
    ax.set_title("SVM-Tron Decision Function Surface")
    plt.show()
