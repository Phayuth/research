import os
import numpy as np
import time
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


qs = dataset[49].detach().cpu().numpy().reshape(-1, 1)
qg = dataset[50].detach().cpu().numpy().reshape(-1, 1)
cmin = np.linalg.norm(qs - qg)
mulp = 1.3
cMAx = cmin * mulp


dd = dataset.detach().cpu().numpy()
print(f"==>> dd: \n{dd}")
xCenter, rotationAxisC, L, cMin = informed_sampling_ellipse(qs, qg, cMAx)
state = isPointinEllipseBulk2(xCenter, rotationAxisC, L, dd)

inpercent = state.sum() / nstore
print(f"==>> Informed sampling ellipse {inpercent * 100:.2f}% of the dataset")
