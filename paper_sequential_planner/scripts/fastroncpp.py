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
        self.fastron.maxUpdates = 5000  # default 1000
        self.fastron.maxSupportPoints = 3000  # default 0, must set by user
        self.fastron.beta = 1000  # default 1

        # tune allowance for active learning with the dataset number
        self.fastron.allowance = 800  # default 800
        self.fastron.kNS = 4  # default 4
        self.fastron.sigma = 0.01  # scale for sampled around support points
        self.fastron.exploitP = 0.5  # default 0.5

    def get_params(self):
        alpha_trained = self.fastron.alpha.copy()
        Gram = self.fastron.G.copy()
        data_support_points = self.fastron.data.copy()

        print(self.fastron.g)
        print(self.fastron.maxUpdates)
        print(self.fastron.maxSupportPoints)
        print(self.fastron.beta)

        print(self.fastron.allowance)
        print(self.fastron.kNS)
        print(self.fastron.sigma)
        print(self.fastron.exploitP)

        return alpha_trained, Gram, data_support_points

    def active_learning(self):
        start_time = time.time()
        self.fastron.activeLearning()
        end_time = time.time()
        print(f"Active Learning Time: {end_time - start_time:.2f} seconds")

    def get_points_to_relabel(self):
        return self.fastron.data.copy()

    def update_labels(self, new_labels):
        start_time = time.time()
        self.fastron.updateLabels(new_labels)
        end_time = time.time()
        print(f"Label Update Time: {end_time - start_time:.2f} seconds")

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
    dataset[:, :2] = dataset[:, :2] / np.pi  # scale to -1, 1
    n_train = 50
    randid = np.random.choice(range(dataset.shape[0]), size=n_train, replace=False)
    dataset_samples = dataset[randid]
    data = np.ascontiguousarray(dataset_samples[:, :2])  # (N, 2), contiguous
    y = np.ascontiguousarray(dataset_samples[:, 2:3])  # (N, 1), contiguous colvec

    def _get_labels_from_dataset(points, dataset):
        labels = []
        for q in points:
            # Find the closest point in the dataset
            dists = np.linalg.norm(dataset[:, :2] - q, axis=1)
            closest_idx = np.argmin(dists)
            labels.append(
                dataset[closest_idx, 2]
            )  # Append the label of the closest point
        return np.array(labels)

    # FASTRON
    # MUST FOLLOW FLOW
    # 1. Initialize FastronCPP with data and labels
    # 2. Update model to train the initial Fastron model
    # 3. Perform active learning to get new points to relabel
    # 4. Update labels for the new points
    fcpp = FastronCPP(data, y)
    fcpp.update_model()
    _, _, data_support_points = fcpp.get_params()
    xtest = dataset[:, :2]
    ytest = dataset[:, 2]
    fcpp.accuracy(xtest, ytest)

    fcpp.fastron.kNS = 10
    fcpp.active_learning()
    tobe_relabel_points = fcpp.get_points_to_relabel()
    new_labels = _get_labels_from_dataset(tobe_relabel_points, dataset)
    new_labels = np.ascontiguousarray(new_labels).reshape(-1, 1)
    fcpp.update_labels(new_labels)
    fcpp.update_model()
    _, _, data_support_points = fcpp.get_params()
    xtest = dataset[:, :2]
    ytest = dataset[:, 2]
    fcpp.accuracy(xtest, ytest)

    # fcpp.fastron.allowance = 100
    # fcpp.fastron.kNS = 10
    # fcpp.active_learning()
    # tobe_relabel_points = fcpp.get_points_to_relabel()
    # tobe_relabel_points = fcpp.get_points_to_relabel()
    # new_labels = _get_labels_from_dataset(tobe_relabel_points, dataset)
    # new_labels = np.ascontiguousarray(new_labels).reshape(-1, 1)
    # fcpp.update_labels(new_labels)
    # fcpp.update_model()
    # _, _, data_support_points = fcpp.get_params()
    # xtest = dataset[:, :2]
    # ytest = dataset[:, 2]
    # fcpp.accuracy(xtest, ytest)

    # fcpp.fastron.allowance = 200
    # fcpp.fastron.kNS = 10
    # fcpp.active_learning()
    # tobe_relabel_points = fcpp.get_points_to_relabel()
    # tobe_relabel_points = fcpp.get_points_to_relabel()
    # new_labels = _get_labels_from_dataset(tobe_relabel_points, dataset)
    # new_labels = np.ascontiguousarray(new_labels).reshape(-1, 1)
    # fcpp.update_labels(new_labels)
    # fcpp.update_model()
    # _, _, data_support_points = fcpp.get_params()
    # xtest = dataset[:, :2]
    # ytest = dataset[:, 2]
    # fcpp.accuracy(xtest, ytest)

    # plot results
    size = 360
    q1 = np.linspace(-1, 1, size)
    q2 = np.linspace(-1, 1, size)
    XX, YY = np.meshgrid(q1, q2)
    Q = np.column_stack([XX.ravel(), YY.ravel()])
    Y = fcpp.fastron.eval(Q)
    datafastron = np.column_stack([Q, Y])
    gt_collision = dataset[dataset[:, 2] == 1][:, :2]
    gt_free = dataset[dataset[:, 2] == -1][:, :2]
    ft_collision = datafastron[datafastron[:, 2] == 1][:, :2]
    train_point = dataset_samples[:, :2]

    xtest = dataset[:, :2]
    ytest = dataset[:, 2]
    fcpp.accuracy(xtest, ytest)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(gt_collision[:, 0], gt_collision[:, 1], "ro", markersize=3)
    ax2.plot(ft_collision[:, 0], ft_collision[:, 1], "bo", markersize=3)
    ax1.plot(
        gt_free[:, 0],
        gt_free[:, 1],
        "co",
        markersize=3,
        alpha=0.5,
        label="Ground truth free",
    )
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
        label="Support points",
    )
    ax2.plot(
        data_support_points[:, 0],
        data_support_points[:, 1],
        "ko",
        markersize=2,
        label="Support points",
    )
    ax2.plot(
        tobe_relabel_points[:, 0],
        tobe_relabel_points[:, 1],
        "mx",
        markersize=5,
        label="To-be-relabel points",
    )
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.show()
