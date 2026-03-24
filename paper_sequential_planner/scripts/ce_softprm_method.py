import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics.pairwise import pairwise_distances
from geometric_ellipse import *
import networkx as nx
from joblib import dump, load

np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))


# Dataset 1 ---------------------------------------------------------------
def is_collision(x):
    R_OBS = 0.8
    return np.linalg.norm(x, axis=1) <= R_OBS


N_TRAIN = 600
dof = 2
X_train = np.random.uniform(-np.pi, np.pi, size=(N_TRAIN, dof))
# y_train = np.where(is_collision(X_train), -1, +1)
y_train = np.where(is_collision(X_train), +1, -1)
# --------------------------------------------------------------------------

# Dataset 2 ----------------------------------------------------------------
dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
N_TRAIN = 1000
samples_id = np.random.choice(range(dataset.shape[0]), size=N_TRAIN, replace=False)
dataset_samples = dataset[samples_id]
X_train = dataset_samples[:, 0:2]
y = dataset_samples[:, 2]
y_train = np.where(y <= 0, -1, +1)  # switch sign
# y_train = np.where(y <= 0, +1, -1)  # switch sign
# --------------------------------------------------------------------------


sigma = 0.5
model = SVC(kernel="rbf", gamma=1 / (2 * sigma**2), C=1.0)
model.fit(X_train, y_train)
fhater = model.decision_function
# N = 10000000, 6dof trained in 3min on laptop

dof = 2
xges = np.array([[0.0] * dof])
xges2 = np.array([[1.0] * dof])
xtdelta = np.array([0.1, 0.1]).reshape(1, -1)  # shape (num_point_query, dof)
# predict label
y_pred = model.predict(xges)
y_pred2 = model.predict(xges2)
print("SVM prediction at xges (0s):", y_pred)
print("SVM prediction at xges2 (1s):", y_pred2)
# predict value
fges = fhater(xges)
fges2 = fhater(xges2)
ftdelta = fhater(xtdelta)
print("SVM decision function at xges (0s):", fges)
print("SVM decision function at xges2 (1s):", fges2)
print("SVM decision function at test point (0.1, 0.1):", ftdelta)

supvecs = model.support_vectors_
supvecs_labels = y_train[model.support_]
supvecs_free = supvecs[supvecs_labels == +1]
supvecs_cols = supvecs[supvecs_labels == -1]
supvecs_fval = fhater(supvecs)
print(f"subvec shape", supvecs.shape)
print("Decision function at support vectors:", supvecs_fval)


def build_adj_matrix(X):
    n = len(X)
    AdjMat = np.full((n, n), np.inf)
    np.fill_diagonal(AdjMat, 0)

    for i in range(n):
        for j in range(i + 1, n):
            print(i, j)
            qs = X[i].reshape(1, -1)
            qg = X[j].reshape(1, -1)
            path = interplolate_line(qs, qg)[:, 0, :]
            score = fhater(path)
            sc = score < 0
            if np.all(sc):
                AdjMat[i, j] = AdjMat[j, i] = 1
    return AdjMat


AdjMat = build_adj_matrix(supvecs)
np.save(os.path.join(rsrc, "softprm_adjmat.npy"), AdjMat)
AdjMat = np.load(os.path.join(rsrc, "softprm_adjmat.npy"))
Md = pairwise_distances(supvecs)
DistAdjMat = Md * AdjMat
print(Md)

ModedDistAdjMat = DistAdjMat.copy()
np.fill_diagonal(ModedDistAdjMat, np.inf)
unconnectednode = np.where(np.isinf(ModedDistAdjMat).all(axis=1))[0]
print("Unconnected nodes:", unconnectednode)


q1 = np.array([-1.0, 2.5])
q2 = np.array([1.0, 2.5])
q3 = np.array([0.15, 0.60])
q4 = np.array([2.5, 1.5])
q5 = np.array([-2.5, -1.5])
q6 = np.array([2.40, -0.4])
q7 = np.array([-2.0, 2.5])
q8 = np.array([1.0, -2.0])
q9 = np.array([-3.0, 0.0])
q10 = np.array([-3.0, 2.5])


# search shortest path
G = nx.from_numpy_array(DistAdjMat)
start_id = 30
goal_id = 60
path = nx.shortest_path(G, start_id, goal_id, weight="weight")
cost = nx.shortest_path_length(G, start_id, goal_id, weight="weight")
pathq = supvecs[path]
print("Path waypoints:", pathq)


fig, ax = plt.subplots()
grid = 200
xs = np.linspace(-np.pi, np.pi, grid)
ys = np.linspace(-np.pi, np.pi, grid)
XX, YY = np.meshgrid(xs, ys)
XY = np.column_stack([XX.ravel(), YY.ravel()])
Z = model.predict(XY).reshape(grid, grid)
Zf = fhater(XY).reshape(grid, grid)
ax.contourf(XX, YY, Z, levels=50, cmap="coolwarm")
ax.contour(XX, YY, Z, levels=[0], colors="black")
# ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
for i in range(AdjMat.shape[0]):
    for j in range(AdjMat.shape[1]):
        if AdjMat[i, j] == 1:
            qs = supvecs[i]
            qg = supvecs[j]
            ax.plot([qs[0], qg[0]], [qs[1], qg[1]], "k--", linewidth=0.5)

ax.plot(
    supvecs_free[:, 0],
    supvecs_free[:, 1],
    "ro",
    markersize=5,
    label="Support Vectors (+1)",
)
ax.plot(
    supvecs_cols[:, 0],
    supvecs_cols[:, 1],
    "bo",
    markersize=5,
    label="Support Vectors (-1)",
)
ax.plot(pathq[:, 0], pathq[:, 1], "g-o", linewidth=2, label="planned path")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_aspect("equal")
ax.set_title("Learned free-space scalar field f̂(x)")
plt.show()


from scipy.spatial import KDTree

kdt = KDTree(supvecs)
print("KDTree built.")

q = np.array([0.0, 0.0]).reshape(1, -1)
d, i = kdt.query(q, k=5)
print("Query point:", q)
print("Nearest neighbors' indices:", i)
print("Nearest neighbors' distances:", d)

fig, ax = plt.subplots()
ax.contourf(XX, YY, Zf, levels=50, cmap="coolwarm")
ax.contour(XX, YY, Zf, levels=[0], colors="black")
ax.plot(
    supvecs_free[:, 0],
    supvecs_free[:, 1],
    "ro",
    markersize=5,
    label="Support Vectors (+1)",
)
ax.plot(
    supvecs_cols[:, 0],
    supvecs_cols[:, 1],
    "bo",
    markersize=5,
    label="Support Vectors (-1)",
)
ax.set_aspect("equal")
ax.set_title("Learned free-space scalar field f̂(x)")
plt.show()

pp = np.linspace(q1, q2, 10)


def mapping(signal, realmin, realmax):
    sigmin = np.min(signal)
    sigmax = np.max(signal)
    scaled = (signal - sigmin) / (sigmax - sigmin)
    mapped = realmin + scaled * (realmax - realmin)
    return mapped


Zf = 1 / (1 + np.exp(-Zf))  # sigmoid
Zf = mapping(Zf, -1, 1)
fig, ax = plt.subplots()
ax.plot([q1[0], q2[0]], [q1[1], q2[1]], "k-", linewidth=2)
ax.plot(pp[:, 0], pp[:, 1], "ro--")
ax.contourf(XX, YY, Zf, levels=50, cmap="coolwarm")
ax.contour(XX, YY, Zf, levels=[0], colors="black")
ax.set_aspect("equal")
ax.set_title("Learned free-space scalar field f̂(x)")
plt.show()
