import os
import time
import numpy as np
import matplotlib.pyplot as plt
from fastronWrapper.fastronWrapper import PyFastron
from sklearn.metrics import accuracy_score

np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]


class FastronCPP:

    def __init__(self, data, y):
        self.data = data
        self.y = y
        self.fastron = PyFastron(self.data)  # where data.shape = (N, d)
        self.fastron.y = self.y  # where y.shape = (N,)
        self.fastron.g = 10
        self.fastron.maxUpdates = 5000
        self.fastron.maxSupportPoints = 3000
        self.fastron.beta = 100

    def get_params(self):
        alpha_trained = self.fastron.alpha
        Gram = self.fastron.G
        data_support_points = self.fastron.data
        return alpha_trained, Gram, data_support_points

    def active_learning(self):
        start_time = time.time()
        self.fastron.activeLearning()
        end_time = time.time()
        print(f"Active Learning Time: {end_time - start_time:.2f} seconds")

    def update_model(self):
        start_time = time.time()
        self.fastron.updateModel()
        end_time = time.time()
        print(f"Model Update Time: {end_time - start_time:.2f} seconds")

    def eval(self, Q):
        return self.fastron.eval(Q)

    def accuracy(self, xtest, ytest):
        ypred = self.fastron.eval(xtest)
        acc = accuracy_score(ytest, ypred)
        print(f"Test accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    # load dataset
    dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
    n_train = 9000
    randid = np.random.choice(range(dataset.shape[0]), size=n_train, replace=False)
    dataset_samples = dataset[randid]
    data = np.ascontiguousarray(dataset_samples[:, :2])  # (N, 2), contiguous
    y = np.ascontiguousarray(dataset_samples[:, 2:3])  # (N, 1), contiguous colvec

    # FASTRON
    fcpp = FastronCPP(data, y)
    fcpp.active_learning()
    fcpp.update_model()
    alpha, Gram, data_support_points = fcpp.get_params()

    # plot results
    size = 360
    q1 = np.linspace(-np.pi, np.pi, size)
    q2 = np.linspace(-np.pi, np.pi, size)
    XX, YY = np.meshgrid(q1, q2)
    Q = np.column_stack([XX.ravel(), YY.ravel()])
    Y = fcpp.fastron.eval(Q)
    datafastron = np.column_stack([Q, Y])
    gt_collision = dataset[dataset[:, 2] == 1][:, :2]
    ft_collision = datafastron[datafastron[:, 2] == 1][:, :2]
    train_point = dataset_samples[:, :2]

    xtest = dataset[:, :2]
    ytest = dataset[:, 2]
    fcpp.accuracy(xtest, ytest)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(gt_collision[:, 0], gt_collision[:, 1], "ro", markersize=3)
    ax2.plot(ft_collision[:, 0], ft_collision[:, 1], "bo", markersize=3)
    ax1.plot(
        train_point[:, 0],
        train_point[:, 1],
        "go",
        markersize=2,
        label="Training samples",
    )
    ax1.plot(
        data_support_points[:, 0],
        data_support_points[:, 1],
        "kx",
        markersize=5,
    )
    ax2.plot(
        data_support_points[:, 0],
        data_support_points[:, 1],
        "ko",
        markersize=2,
    )
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-np.pi, np.pi)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-np.pi, np.pi)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.show()
