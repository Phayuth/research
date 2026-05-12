import os
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
from env_planarrr import PlanarRR, RobotScene
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import cudf
import cugraph

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robot = PlanarRR()
scene = RobotScene(robot, None)

# n = 2000
# Qsamples = np.random.rand(n, 2) * 2 * np.pi - np.pi
# col_states = np.array([scene.collision_checker(q) for q in Qsamples])
# xcol = Qsamples[col_states]
# xfre = Qsamples[~col_states]


# X = np.asarray(xfre, dtype=np.float32)

# k = 16
# knn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
# knn.fit(X)
# dist, nbr = knn.kneighbors(X)
# N = X.shape[0]

# indptr = [0]
# indices = []
# weights = []
# for i in range(N):

#     for j, d in zip(nbr[i], dist[i]):

#         if i == j:
#             continue

#         indices.append(j)
#         weights.append(float(d))

#     indptr.append(len(indices))

# indptr = np.asarray(indptr, dtype=np.int64)
# indices = np.asarray(indices, dtype=np.int32)
# weights = np.asarray(weights, dtype=np.float32)

# with open("graph.txt", "w") as f:
#     N = len(indptr) - 1
#     for src in range(N):
#         start = indptr[src]
#         end = indptr[src + 1]
#         for k in range(start, end):
#             dst = indices[k]
#             srci = int(src)
#             dsti = int(dst)
#             w = float(weights[k])
#             w_int = max(1, int(w * (1 << 18)))
#             f.write(f"{srci} {dsti} {w_int}\n")

# df = cudf.read_csv(
#     "graph.txt", sep=" ", header=None, names=["src", "dst", "weight"]
# )

raise
src_array = []
dst_array = []
weight_array = []
N = len(indptr) - 1
for src in range(N):
    start = indptr[src]
    end = indptr[src + 1]
    for k in range(start, end):
        dst = indices[k]
        w = weights[k]
        src_array.append(src)
        dst_array.append(dst)
        weight_array.append(w)

df = cudf.DataFrame(
    {
        "src": src_array,
        "dst": dst_array,
        "weight": weight_array,
    }
)

G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
root = 0
dist = cugraph.sssp(G, source=root)
print(dist.head())

pdf = dist.to_pandas()
pred = dict(zip(pdf["vertex"], pdf["predecessor"]))
target = 200
path = []
cur = target
while cur != -1:
    path.append(cur)
    if cur == 0:
        break
    cur = pred[cur]
path.reverse()
print(path)

qpath = xfre[path]
print(f"==>> qpath: \n{qpath}")
# qs = [0.0, 0.0]
# qg = [np.pi / 2, np.pi / 2]
cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
fig, ax = plt.subplots()
ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
ax.plot(xfre[root, 0], xfre[root, 1], "go", markersize=10, label="Start")
ax.plot(xfre[target, 0], xfre[target, 1], "bo", markersize=10, label="Goal")
ax.plot(xfre[:, 0], xfre[:, 1], "k.", markersize=1, label="Free Points")
ax.plot(xcol[:, 0], xcol[:, 1], "ko", markersize=1, label="Collision Points")
ax.plot(qpath[:, 0], qpath[:, 1], "cx", markersize=5, label="Planned Path")
# for i in range(N):
#     for j in range(indptr[i], indptr[i + 1]):
#         nbr_idx = indices[j]
#         w = weights[j]
#         ax.plot(
#             [X[i, 0], X[nbr_idx, 0]],
#             [X[i, 1], X[nbr_idx, 1]],
#             "c-",
#             linewidth=0.5,
#             alpha=0.5,
#         )
ax.legend()
ax.set_aspect("equal", "box")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.grid(True)
plt.show()
