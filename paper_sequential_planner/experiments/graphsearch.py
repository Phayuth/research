import os
import numpy as np
import tqdm
import torch
from paper_sequential_planner.scripts.geometric_ellipse import *
from paper_sequential_planner.experiments.env_ur5e_sphere import (
    RobotUR5eKin,
    SceneUR5eSpherized,
    device,
    pick_task_poses,
)

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]

robkin = RobotUR5eKin()
scene = SceneUR5eSpherized()
planner = None  # placeholder for planner instance, not implemented yet

nstore = 1000000
dof = 6
dataset = torch.empty(nstore, dof, dtype=torch.float16, device=device)
dataset_y = torch.empty(nstore, dtype=torch.bool, device=device)
batch = 100000
it = nstore // batch
for i in tqdm.tqdm(range(it)):
    start = i * batch
    end = start + batch
    dataset[start:end, :dof] = (
        torch.rand(batch, dof, dtype=torch.float16, device=device) * torch.pi
        - torch.pi / 2
    )
    col_states = scene.collision_check(dataset[start:end, :dof])
    dataset_y[start:end] = col_states

collision_rate = dataset_y.sum().item() / nstore
print(f"==>> Dataset collision rate: {collision_rate * 100:.2f}%")

xfree = dataset[~dataset_y]
print(f"==>> xfree.shape: \n{xfree.shape}")
print(f"dataset is on device: {dataset.device}")


# # limit search space per query
# qs = dataset[49].detach().cpu().numpy().reshape(-1, 1)
# qg = dataset[50].detach().cpu().numpy().reshape(-1, 1)
# cmin = np.linalg.norm(qs - qg)
# mulp = 1.3
# cMAx = cmin * mulp
# dd = dataset.detach().cpu().numpy()
# print(f"==>> dd: \n{dd}")
# xCenter, rotationAxisC, L, cMin = informed_sampling_ellipse(qs, qg, cMAx)
# state = isPointinEllipseBulk2(xCenter, rotationAxisC, L, dd)
# inpercent = state.sum() / nstore
# print(f"==>> Informed sampling ellipse {inpercent * 100:.2f}% of the dataset")


import faiss
res = faiss.StandardGpuResources()  # use a single GPU
index_flat = faiss.IndexFlatL2(dof)  # build a flat (CPU) index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index_flat.add(dataset.cpu().numpy())  # add vectors to the index
print(gpu_index_flat.ntotal)
k = 16
D, I = gpu_index_flat.search(xfree.cpu().numpy(), k)  # actual search

# ## Using an IVF index
# nlist = 100
# quantizer = faiss.IndexFlatL2(d)  # the other index
# index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# # here we specify METRIC_L2, by default it performs inner-product search
# # make it an IVF GPU index
# gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)
# assert not gpu_index_ivf.is_trained
# gpu_index_ivf.train(xb)  # add vectors to the index
# assert gpu_index_ivf.is_trained
# gpu_index_ivf.add(xb)  # add vectors to the index
# print(gpu_index_ivf.ntotal)
# k = 4  # we want to see 4 nearest neighbors
# D, I = gpu_index_ivf.search(xq, k)  # actual search
