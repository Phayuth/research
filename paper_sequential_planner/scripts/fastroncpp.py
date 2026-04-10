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


def iterative_2d():
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
        "r+",
        markersize=10,
        label="Support points",
    )
    ax2.plot(
        data_support_points[:, 0],
        data_support_points[:, 1],
        "r+",
        markersize=10,
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


def oneshot_2d():
    # load dataset
    dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
    dataset[:, :2] = dataset[:, :2] / np.pi  # scale to -1, 1
    n_train = 9000
    randid = np.random.choice(range(dataset.shape[0]), size=n_train, replace=False)
    dataset_samples = dataset[randid]
    data = np.ascontiguousarray(dataset_samples[:, :2])  # (N, 2), contiguous
    y = np.ascontiguousarray(dataset_samples[:, 2:3])  # (N, 1), contiguous colvec

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
    np.save(os.path.join(rsrc, "ftron_2d_support_points.npy"), data_support_points)

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
        "r+",
        markersize=10,
        label="Support points",
    )
    ax2.plot(
        data_support_points[:, 0],
        data_support_points[:, 1],
        "r+",
        markersize=10,
        label="Support points",
    )
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.show()


def oneshot_3d():
    import trimesh

    # load dataset
    dataset = np.load(os.path.join(rsrc, "spatial3r_cspace.npy"))
    dataset[:, 3] = np.where(dataset[:, 3] == 0, -1, dataset[:, 3])
    N_TRAIN = 10000
    randid = np.random.choice(range(dataset.shape[0]), size=N_TRAIN, replace=False)
    dataset_samples = dataset[randid]
    data = np.ascontiguousarray(dataset_samples[:, :3])  # (N, 3), contiguous
    y = np.ascontiguousarray(
        dataset_samples[:, 3:4]
    )  # (N, 1), contiguous col vector

    # FASTRON
    fcpp = FastronCPP(data, y)
    fcpp.active_learning()
    fcpp.update_model()
    alpha, Gram, data_support_points = fcpp.get_params()

    # # plot
    # size = 360
    # q1 = np.linspace(-np.pi, np.pi, size)
    # q2 = np.linspace(-np.pi, np.pi, size)
    # q3 = np.linspace(-np.pi, np.pi, size)
    # XX, YY, ZZ = np.meshgrid(q1, q2, q3)
    # Q = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    # Y = fcpp.fastron.eval(Q)
    # datafastron = np.column_stack([Q, Y])
    # gt_collision = dataset[dataset[:, 3] == 1][:, :3]
    # ft_collision = datafastron[datafastron[:, 3] == 1][:, :3]
    # train_point = dataset_samples[:, :3]

    # xtest = dataset[:, :3]
    # ytest = dataset[:, 3]
    # fcpp.accuracy(xtest, ytest)  #  88.82%

    # Qfree = dataset[dataset[:, -1] == 0][:, :3]
    # Qcoll = dataset[dataset[:, -1] == 1][:, :3]

    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=np.pi)
    box = trimesh.creation.box(extents=(2 * np.pi, 2 * np.pi, 2 * np.pi))
    box.visual.face_colors = [100, 150, 255, 40]
    pp = trimesh.load_path(box.vertices[box.edges_unique])
    scene.add_geometry(box)
    scene.add_geometry(axis)
    scene.add_geometry(pp)
    # Qr = trimesh.points.PointCloud(Qcoll, colors=[255, 0, 0, 255])
    # scene.add_geometry(Qr)
    # Qfastron = trimesh.points.PointCloud(ft_collision, colors=[0, 0, 255, 255])
    # scene.add_geometry(Qfastron)
    # Qtrain = trimesh.points.PointCloud(train_point, colors=[0, 255, 0, 255])
    # scene.add_geometry(Qtrain)
    Qsupport = trimesh.points.PointCloud(
        data_support_points, colors=[0, 0, 0, 255]
    )
    scene.add_geometry(Qsupport)
    scene.show()


def support_points_normal_vector():

    def estimate_outward_normals(support_points, obstacle_points, k_neighbors=20):
        """Estimate outward normals for support points using local obstacle PCA."""
        if support_points.ndim != 2 or support_points.shape[1] != 2:
            raise ValueError("support_points must have shape (N, 2)")
        if obstacle_points.ndim != 2 or obstacle_points.shape[1] != 2:
            raise ValueError("obstacle_points must have shape (M, 2)")
        if obstacle_points.shape[0] < 2:
            raise ValueError(
                "Need at least two obstacle points for normal estimation"
            )

        k = int(np.clip(k_neighbors, 2, obstacle_points.shape[0]))
        normals = np.zeros_like(support_points, dtype=float)

        for i, sp in enumerate(support_points):
            diffs = obstacle_points - sp
            d2 = np.einsum("ij,ij->i", diffs, diffs)
            nn_idx = np.argpartition(d2, k - 1)[:k]
            local_obs = obstacle_points[nn_idx]

            centroid = np.mean(local_obs, axis=0)
            centered = local_obs - centroid
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, np.argmin(eigvals)]

            # Orient normal away from nearby obstacle samples.
            away = sp - centroid
            if np.dot(normal, away) < 0.0:
                normal = -normal

            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                normal = away
                norm = np.linalg.norm(normal)
            if norm < 1e-12:
                normal = np.array([1.0, 0.0])
                norm = 1.0

            normals[i] = normal / norm

        return normals

    cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles_extended.npy"))
    ftronsp = np.load(os.path.join(rsrc, "ftron_2d_support_points.npy"))
    ftronsp = ftronsp * np.pi  # rescale to [-pi, pi]
    nft = ftronsp.shape[0]
    normals = estimate_outward_normals(
        support_points=ftronsp,
        obstacle_points=cspace_obs,
        k_neighbors=20,
    )

    qa = np.array([0.0, 0.0])
    qb = np.array([2.0, 1.0])
    path = np.linspace(qa, qb, num=nft)

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        cspace_obs[:, 0],
        cspace_obs[:, 1],
        "x",
        color="red",
        label="C-space Obstacles",
    )
    ax.plot(
        ftronsp[:, 0],
        ftronsp[:, 1],
        "+",
        color="orange",
        label="Fastron Support Points",
    )
    ax.quiver(
        ftronsp[:, 0],
        ftronsp[:, 1],
        normals[:, 0],
        normals[:, 1],
        angles="xy",
        scale_units="xy",
        scale=8.0,
        color="black",
        width=0.003,
        alpha=0.7,
        label="Estimated Outward Normals",
    )
    ax.plot(qa[0], qa[1], marker="o", color="blue", label="qa")
    ax.plot(qb[0], qb[1], marker="o", color="green", label="qb")
    ax.plot(
        path[:, 0], path[:, 1], color="purple", marker=".", label="Linear Path"
    )

    ax.set_aspect("equal", "box")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # iterative_2d()
    oneshot_2d()
