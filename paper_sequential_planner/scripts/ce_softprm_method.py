import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from geometric_ellipse import *


np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))


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


# Dataset 1 ---------------------------------------------------------------
def is_collision(x):
    R_OBS = 0.8
    return np.linalg.norm(x, axis=1) <= R_OBS


# N_TRAIN = 600
# dof = 2
# X_train = np.random.uniform(-np.pi, np.pi, size=(N_TRAIN, dof))
# # y_train = np.where(is_collision(X_train), -1, +1)
# y_train = np.where(is_collision(X_train), +1, -1)
# xfreegarantee = np.array([1, 1]).reshape(1, -1)
# --------------------------------------------------------------------------

# Dataset 2 ----------------------------------------------------------------
dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
N_TRAIN = 1000
samples_id = np.random.choice(range(dataset.shape[0]), size=N_TRAIN, replace=False)
dataset_samples = dataset[samples_id]
X_train = dataset_samples[:, 0:2]
y = dataset_samples[:, 2]
# y_train = np.where(y <= 0, -1, +1)  # switch sign
y_train = np.where(y <= 0, +1, -1)  # switch sign
xfreegarantee = np.array([0.0, 0.0]).reshape(1, -1)
# --------------------------------------------------------------------------


sigma = 0.5
model = SVC(kernel="rbf", gamma=1 / (2 * sigma**2), C=1.0)
ynnn_train = np.where(y_train == -1, 1, -1)

model.fit(X_train, ynnn_train)
fhater = model.decision_function
# N = 10000000, 6dof trained in 3min on laptop

dof = 2
xges = np.array([[0.0] * dof])
xges2 = np.array([[1.0] * dof])
xtdelta = np.array([[0.1, 0.1]])
y_pred = model.predict(xges)
y_pred2 = model.predict(xges2)
fges = fhater(xges)
fges2 = fhater(xges2)
ftdelta = fhater(xtdelta)
print(xges.shape)
print("SVM prediction at xges (0s):", y_pred)
print("SVM prediction at xges2 (1s):", y_pred2)
print("SVM decision function at xges (0s):", fges)
print("SVM decision function at xges2 (1s):", fges2)
print("SVM decision function at test point (0.1, 0.1):", ftdelta)

supvecs = model.support_vectors_
supvecs_labels = y_train[model.support_]
supvecs_free = supvecs[supvecs_labels == +1]
supvecs_cols = supvecs[supvecs_labels == -1]


grid = 200
xs = np.linspace(-np.pi, np.pi, grid)
ys = np.linspace(-np.pi, np.pi, grid)
XX, YY = np.meshgrid(xs, ys)
XY = np.column_stack([XX.ravel(), YY.ravel()])
Z = model.predict(XY).reshape(grid, grid)


ffff = fhater(supvecs)
print("Decision function at support vectors:", ffff)
print(supvecs.shape)
Md = pairwise_distances(supvecs)
print(Md.shape)

Qqqq = np.array([2.5, 2.5]).reshape(1, -1)
fQqqq = fhater(Qqqq)
print("Decision function at Qqqq (2.5,2.5):", fQqqq)

AdjMat = np.zeros(Md.shape)
for i in range(Md.shape[0]):
    for j in range(Md.shape[1]):
        if i != j:
            print(i, j)
            qs = supvecs[i].reshape(1, -1)
            qg = supvecs[j].reshape(1, -1)
            path = interplolate_line(qs, qg)[:, 0, :]
            score = [fhater(pt.reshape(1, -1)).item() for pt in path]
            sc = np.array(score) < 0
            if np.all(sc):
                AdjMat[i, j] = 1
            else:
                AdjMat[i, j] = 0


# --------------------------------------------------------------------


fig, ax = plt.subplots()
ax.contourf(XX, YY, Z, levels=50, cmap="coolwarm")
ax.contour(XX, YY, Z, levels=[0], colors="black")
# ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
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
for i in range(AdjMat.shape[0]):
    for j in range(AdjMat.shape[1]):
        if AdjMat[i, j] == 1:
            qs = supvecs[i]
            qg = supvecs[j]
            ax.plot([qs[0], qg[0]], [qs[1], qg[1]], "k--", linewidth=0.5)
ax.set_aspect("equal")
ax.set_title("Learned free-space scalar field f̂(x)")
plt.show()


raise SystemExit

grid = 200
xs = np.linspace(-np.pi, np.pi, grid)
ys = np.linspace(-np.pi, np.pi, grid)
XX, YY = np.meshgrid(xs, ys)
XY = np.column_stack([XX.ravel(), YY.ravel()])
Z = fhater(XY).reshape(grid, grid)

fig, ax = plt.subplots()
ax.contourf(XX, YY, Z, levels=50, cmap="coolwarm")
ax.contour(XX, YY, Z, levels=[0], colors="black")
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
fp = fhater(pp)
print("SVM decision function along line from q1 to q2:", fp)


def mapping(signal, realmin, realmax):
    sigmin = np.min(signal)
    sigmax = np.max(signal)
    scaled = (signal - sigmin) / (sigmax - sigmin)
    mapped = realmin + scaled * (realmax - realmin)
    return mapped


grid = 200
xs = np.linspace(-np.pi, np.pi, grid)
ys = np.linspace(-np.pi, np.pi, grid)
XX, YY = np.meshgrid(xs, ys)
XY = np.column_stack([XX.ravel(), YY.ravel()])
Z = fhater(XY).reshape(grid, grid)
Z = 1 / (1 + np.exp(-Z))  # sigmoid
Z = mapping(Z, -1, 1)
print(np.max(Z), np.min(Z))

fig, ax = plt.subplots()
ax.plot([q1[0], q2[0]], [q1[1], q2[1]], "k-", linewidth=2)
ax.plot(pp[:, 0], pp[:, 1], "ro--")
ax.contourf(XX, YY, Z, levels=50, cmap="coolwarm")
ax.contour(XX, YY, Z, levels=[0], colors="black")
ax.set_aspect("equal")
ax.set_title("Learned free-space scalar field f̂(x)")
plt.show()
