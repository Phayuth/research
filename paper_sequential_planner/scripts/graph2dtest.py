import os
import numpy as np
import tqdm
import torch
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


def sklearn_knn(xfre):
    X = np.asarray(xfre, dtype=np.float32)
    k = 16
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    knn.fit(X)
    dist, nbr = knn.kneighbors(X)
    print(f"==>> dist: \n{dist}")
    print(f"==>> nbr: \n{nbr}")
    N = X.shape[0]
    return knn, dist, nbr, N, k


knn, dist, nbr, N, k = sklearn_knn(xfre)
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

cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
fig, ax = plt.subplots()
ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
ax.plot(xfre[root, 0], xfre[root, 1], "go", markersize=10, label="Start")
ax.plot(xfre[target, 0], xfre[target, 1], "bo", markersize=10, label="Goal")
ax.plot(xfre[:, 0], xfre[:, 1], "k.", markersize=1, label="Free Points")
ax.plot(xcol[:, 0], xcol[:, 1], "ko", markersize=1, label="Collision Points")
ax.plot(qpath[:, 0], qpath[:, 1], "cx--", markersize=5, label="Planned Path")
# for i in range(src_array.shape[0]):
#     src = src_array[i]
#     dst = dst_array[i]
#     weight = weight_array[i]
#     ax.plot(
#         [xfre[src, 0], xfre[dst, 0]],
#         [xfre[src, 1], xfre[dst, 1]],
#         "k-",
#         alpha=0.01,
#     )
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

# from paper_sequential_planner.experiments.utilio import extract_paths

# tsv_file = "paper_sequential_planner/experiments/combined_paths.tsv"
# paths = extract_paths(tsv_file)
# print(f"==>> paths: \n{paths}")

# fig, ax = plt.subplots()
# ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
# ax.plot(xfre[:, 0], xfre[:, 1], "k.", markersize=1, label="Free Points")
# ax.plot(xcol[:, 0], xcol[:, 1], "ko", markersize=1, label="Collision Points")
# ax.plot(Qs[:, 0], Qs[:, 1], "go", markersize=10, label="Starts")
# ax.plot(Qg[:, 0], Qg[:, 1], "bo", markersize=10, label="Goals")
# for i in range(len(paths)):
#     path = paths[i]
#     if path is not None:
#         qpath = xfre[path]
#         ax.plot(
#             qpath[:, 0],
#             qpath[:, 1],
#             "cx--",
#             markersize=5,
#             label=f"Planned Path {i}",
#         )
# ax.set_aspect("equal", "box")
# ax.set_xlim(-np.pi, np.pi)
# ax.set_ylim(-np.pi, np.pi)
# ax.grid(True)
# ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.show()


import faiss
dof = 2
res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatL2(dof)
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index_flat.add(xfre.astype(np.float32))

k = 16
D, I = gpu_index_flat.search(xfre.astype(np.float32), k)
print(f"==>> D: \n{D}")
print(f"==>> I: \n{I}")


