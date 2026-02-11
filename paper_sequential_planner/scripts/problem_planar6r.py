import os
import numpy as np
from shapely.geometry import LineString, box, Polygon, MultiPolygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def forward_kinematics_vectorized(joint_angles):
    link_lengths = np.ones(6)  # Assuming all link lengths are 1 unit
    n, num_joints = joint_angles.shape
    cumulative_angles = np.cumsum(joint_angles, axis=1)
    x_displacements = link_lengths * np.cos(cumulative_angles)
    y_displacements = link_lengths * np.sin(cumulative_angles)
    displacements = np.stack((x_displacements, y_displacements), axis=-1)
    positions = np.cumsum(displacements, axis=1)
    base = np.zeros((n, 1, 2))  # Shape (n, 1, 2), to represent the base at (0, 0)
    result = np.concatenate([base, positions], axis=1)  # Shape (n, 7, 2)
    return result


# o1 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])
# o2 = Polygon([(-4, 3), (-3, 3), (-3, 4), (-4, 4)])
# oo = MultiPolygon([o1, o2])
o1 = Polygon([(-6,-6), (-2,-6), (-2,6), (-6,6)])
oo = MultiPolygon([o1])
clearence = 0.1
rb = 0.2


def get_dmin(xy_coordinates):
    dmin = LineString(xy_coordinates).buffer(rb).distance(oo)
    return dmin


def compute_obstacle_cost_one_traj(traj):
    xy_coordinates = forward_kinematics_vectorized(traj)
    armcols = [LineString(pp).buffer(rb) for pp in xy_coordinates]
    dmin = np.array([armcol.distance(oo) for armcol in armcols])
    cost = np.maximum(clearence + rb - dmin, 0)
    return np.sum(cost)


def collision_check(q):
    xy_coordinates = forward_kinematics_vectorized(q[np.newaxis, :])[0]
    dmin = get_dmin(xy_coordinates)
    if dmin <= 0:
        return True
    else:
        return False


def show_env(q):
    xy_coordinates = forward_kinematics_vectorized(q[np.newaxis, :])[0]
    armcols = LineString(xy_coordinates).buffer(rb)

    dmin = get_dmin(xy_coordinates)
    armcols_nearest, oo_nearest = nearest_points(armcols, oo)

    fig, ax = plt.subplots()
    ax.plot(xy_coordinates[:, 0], xy_coordinates[:, 1], "-o", color="blue")
    x, y = armcols.exterior.xy
    ax.fill(x, y, alpha=0.5, fc="green", ec="black")
    for poly in oo.geoms:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc="red", ec="black")
    ax.plot(
        [oo_nearest.x, armcols_nearest.x],
        [oo_nearest.y, armcols_nearest.y],
        "o--",
        color="purple",
        markersize=12,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.grid()
    plt.show()


def sample_uniform(n, limits):
    d = len(limits)
    X = np.random.rand(n, d)
    for i, (lo, hi) in enumerate(limits):
        X[:, i] = lo + X[:, i] * (hi - lo)
    return X


def label_points(X):
    y = np.array([collision_check(q) for q in X], dtype=int)
    return y


def refine_boundary(X, y, limits, n_new=5000, eps=0.02):
    d = X.shape[1]
    new_pts = []

    idx = np.random.choice(len(X), size=min(len(X), n_new), replace=False)

    for i in idx:
        q = X[i]
        for _ in range(3):  # few local probes
            dq = eps * np.random.randn(d)
            qn = q + dq

            # clip to limits
            for j, (lo, hi) in enumerate(limits):
                qn[j] = np.clip(qn[j], lo, hi)

            if collision_check(qn) != bool(y[i]):
                new_pts.append(qn)

    return np.array(new_pts)


if __name__ == "__main__":
    num_samples = 20
    q1 = np.linspace(-np.pi, np.pi, num_samples)
    q2 = np.linspace(-np.pi, np.pi, num_samples)
    q3 = np.linspace(-np.pi, np.pi, num_samples)
    q4 = np.linspace(-np.pi, np.pi, num_samples)
    q5 = np.linspace(-np.pi, np.pi, num_samples)
    q6 = np.linspace(-np.pi, np.pi, num_samples)

    # Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
    # data = np.column_stack([Q1.ravel(), Q2.ravel()])
    # print(data)
    # print(data.shape)  # (400, 2)

    # Q1, Q2, Q3, Q4, Q5, Q6 = np.meshgrid(q1, q2, q3, q4, q5, q6, indexing="ij")
    # data = np.column_stack(
    #     [Q1.ravel(), Q2.ravel(), Q3.ravel(), Q4.ravel(), Q5.ravel(), Q6.ravel()]
    # )
    # print(data)
    # print(data.shape)  # (64_000_000, 6)

    q = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
    show_env(q)

    joint_limits = [(-np.pi, np.pi)] * 6
    X = sample_uniform(100000, joint_limits)
    y = label_points(X)
    Xb = refine_boundary(X, y, joint_limits, n_new=8000)
    yb = label_points(Xb)
    X_train = np.vstack([X, Xb])
    y_train = np.hstack([y, yb])

    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")

    class1 = np.sum(y_train == 1)
    class0 = np.sum(y_train == 0)
    print(f"Class 1 (in-collision) samples: {class1}")
    print(f"Class 0 (collision-free) samples: {class0}")

    # raise
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    sigma = 0.5
    model = SVC(kernel="rbf", gamma=1 / (2 * sigma**2), C=1.0)
    model.fit(X_train, y_train)
    fhater = model.decision_function
    print("Training completed.")
    from joblib import dump, load

    # dump(model, os.path.join(rsrc, "svm_6dof_model.joblib"))
    model = load(os.path.join(rsrc, "svm_6dof_model.joblib"))

    # Evaluate test accuracy
    # xtest = sample_uniform(100000, joint_limits)
    # ytest = label_points(xtest)
    # ypred = model.predict(xtest)
    # acc = accuracy_score(ytest, ypred)
    # print(f"Test accuracy: {acc*100:.2f}%")

    supvecs = model.support_vectors_.shape
    print(f"Support vectors shape: {supvecs}")

    qrand = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)
    pred = model.predict(qrand)
    print("Random query:", qrand)
    print("SVM prediction at random query:", pred)
    qrand = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)
    pred = model.predict(qrand)
    print("Random query:", qrand)
    print("SVM prediction at random query:", pred)

    show_env(qrand[0])
