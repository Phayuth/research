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

n = 2000
Qsamples = np.random.rand(n, 2) * 2 * np.pi - np.pi
col_states = np.array([scene.collision_checker(q) for q in Qsamples])
xcol = Qsamples[col_states]
xfre = Qsamples[~col_states]

X = np.asarray(xfre, dtype=np.float32)
k = 16
knn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
knn.fit(X)
dist, nbr = knn.kneighbors(X)
N = X.shape[0]

src_array = np.repeat(np.arange(N), k)
dst_array = nbr.flatten()
weight_array = dist.flatten()
df = cudf.DataFrame(
    {
        "src": src_array,
        "dst": dst_array,
        "weight": weight_array,
    }
)
G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")

qs = np.array([-3.0, 3.0])
qe = np.array([3.0, -3.0])
dist, nbr = knn.kneighbors(qs.reshape(1, -1))
root = nbr[0][0]
dist, nbr = knn.kneighbors(qe.reshape(1, -1))
target = nbr[0][0]
print(f"==>> root: {root}, target: {target}")

dist = cugraph.sssp(G, source=root)
pdf = dist.to_pandas()
pred = dict(zip(pdf["vertex"], pdf["predecessor"]))
path = []
cur = target
while cur != -1:
    path.append(cur)
    if cur == 0:
        break
    cur = pred[cur]
path.reverse()
qpath = xfre[path]

pathid = [
    3,
    121,
    592,
    873,
    157,
    676,
    1132,
    828,
    844,
    383,
    1374,
    923,
    703,
    1111,
    619,
    1356,
    1416,
    1268,
    1048,
    218,
    586,
    727,
    581,
    612,
    334,
    911,
    1183,
    597,
    1451,
    733,
    1441,
    860,
    161,
    1237,
    1532,
]
pathsss = xfre[pathid]

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
fig, ax = plt.subplots()
ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
ax.plot(xfre[root, 0], xfre[root, 1], "go", markersize=10, label="Start")
ax.plot(xfre[target, 0], xfre[target, 1], "bo", markersize=10, label="Goal")
ax.plot(xfre[:, 0], xfre[:, 1], "k.", markersize=1, label="Free Points")
ax.plot(xcol[:, 0], xcol[:, 1], "ko", markersize=1, label="Collision Points")
ax.plot(qpath[:, 0], qpath[:, 1], "cx--", markersize=5, label="Planned Path")
ax.plot(pathsss[:, 0], pathsss[:, 1], "mx--", markersize=5, label="Sample Path")
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


Qs = np.array(
    [
        [-3, 3],
        [-3, 0],
        [-3, -1],
        [-3, -2],
        [-3, -3],
    ]
)
Qg = np.array(
    [
        [3, -3],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
)

roots = []
targets = []
for qs in Qs:
    dist, nbr = knn.kneighbors(qs.reshape(1, -1))
    root = nbr[0][0]
    roots.append(root.item())
for qg in Qg:
    dist, nbr = knn.kneighbors(qg.reshape(1, -1))
    target = nbr[0][0]
    targets.append(target.item())

print(f"==>> roots: {roots}")
print(f"==>> targets: {targets}")

from paper_sequential_planner.experiments.utilio import extract_paths

tsv_file =  "paper_sequential_planner/experiments/combined_paths.tsv"
paths = extract_paths(tsv_file)
print(f"==>> paths: \n{paths}")

fig, ax = plt.subplots()
ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
ax.plot(xfre[:, 0], xfre[:, 1], "k.", markersize=1, label="Free Points")
ax.plot(xcol[:, 0], xcol[:, 1], "ko", markersize=1, label="Collision Points")
ax.plot(Qs[:, 0], Qs[:, 1], "go", markersize=10, label="Starts")
ax.plot(Qg[:, 0], Qg[:, 1], "bo", markersize=10, label="Goals")
for i in range(len(paths)):
    path = paths[i]
    if path is not None:
        qpath = xfre[path]
        ax.plot(qpath[:, 0], qpath[:, 1], "cx--", markersize=5, label=f"Planned Path {i}")
ax.set_aspect("equal", "box")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.grid(True)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()