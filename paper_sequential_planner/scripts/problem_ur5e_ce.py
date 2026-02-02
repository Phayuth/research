import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics.pairwise import pairwise_distances
from geometric_ellipse import *
import networkx as nx
from sklearn.metrics import accuracy_score
from joblib import dump, load

np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]


# Dataset 1 ---------------------------------------------------------------
def is_collision(x):
    R_OBS = 0.8
    return np.linalg.norm(x, axis=1) <= R_OBS


N_TRAIN = 10000000  # N = 10000000, 6dof trained in 3mn50s on laptop
dof = 6
X_train = np.random.uniform(-np.pi, np.pi, size=(N_TRAIN, dof))
# y_train = np.where(is_collision(X_train), -1, +1)  # -1: collision, +1: free
y_train = np.where(is_collision(X_train), +1, -1)  # -1: free, +1: collision
# --------------------------------------------------------------------------


sigma = 0.5
model = SVC(kernel="rbf", gamma=1 / (2 * sigma**2), C=1.0)
# model.fit(X_train, y_train)

# dump(model, os.path.join(rsrc, "svm_6dof_model.joblib"))
model = load(os.path.join(rsrc, "svm_6dof_model.joblib"))
fhater = model.decision_function

X_test = np.random.uniform(-np.pi, np.pi, size=(10000, dof))
y_test = np.where(is_collision(X_test), +1, -1)
y_pred = model.predict(X_test)
accscore = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accscore*100:.2f}%")

# raise
x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
pred = model.predict(x0)
predf = fhater(x0)
print(f"Prediction for {x0}: {pred}")
print(f"Decision function value for {x0}: {predf}")

x1 = np.array([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])
pred1 = model.predict(x1)
predf1 = fhater(x1)
print(f"Prediction for {x1}: {pred1}")
print(f"Decision function value for {x1}: {predf1}")


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
            qs = X[i].reshape(1, -1)
            qg = X[j].reshape(1, -1)
            path = interplolate_line(qs, qg)[:, 0, :]
            score = fhater(path)
            sc = score < 0
            if np.all(sc):
                AdjMat[i, j] = AdjMat[j, i] = 1
    return AdjMat


# AdjMat = build_adj_matrix(supvecs)
# np.save(os.path.join(rsrc, "softprm_adjmat_ur5e.npy"), AdjMat)
AdjMat = np.load(os.path.join(rsrc, "softprm_adjmat_ur5e.npy"))
Md = pairwise_distances(supvecs)
DistAdjMat = Md * AdjMat
print(Md)

ModedDistAdjMat = DistAdjMat.copy()
np.fill_diagonal(ModedDistAdjMat, np.inf)
unconnectednode = np.where(np.isinf(ModedDistAdjMat).all(axis=1))[0]
print("Unconnected nodes:", unconnectednode)


# start = np.array([-2.5, -2.5, -2.5, -2.5, -2.5, -2.5])
# goal = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])

# search shortest path
G = nx.from_numpy_array(DistAdjMat)
start_id = 0
goal_id = 204
path = nx.shortest_path(G, start_id, goal_id, weight="weight")
cost = nx.shortest_path_length(G, start_id, goal_id, weight="weight")
pathq = supvecs[path]
print("Path waypoints:", pathq)

qs = supvecs[start_id].reshape(1, -1)
qg = supvecs[goal_id].reshape(1, -1)
pathstraight = interplolate_line(qs, qg, n=4)[:, 0, :]
print(pathstraight)


fig, ax = plt.subplots(6, 1, figsize=(8, 12))
for i in range(dof):
    ax[i].plot(pathq[:, i], "o-", label="est col-free path")
    ax[i].plot(pathstraight[:, i], "x--", label="Straight line")
    ax[i].set_ylabel(f"Joint {i+1} (rad)")
ax[-1].set_xlabel("Waypoint index")
ax[0].legend()
plt.tight_layout()
plt.show()