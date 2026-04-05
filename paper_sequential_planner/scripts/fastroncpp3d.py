import os
import numpy as np
import trimesh
from fastroncpp import FastronCPP

np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]

# load dataset
dataset = np.load(os.path.join(rsrc, "spatial3r_cspace.npy"))
dataset[:, 3] = np.where(dataset[:, 3] == 0, -1, dataset[:, 3])
N_TRAIN = 10000
randid = np.random.choice(range(dataset.shape[0]), size=N_TRAIN, replace=False)
dataset_samples = dataset[randid]
data = np.ascontiguousarray(dataset_samples[:, :3])  # (N, 3), contiguous
y = np.ascontiguousarray(dataset_samples[:, 3:4])  # (N, 1), contiguous col vector

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
Qsupport = trimesh.points.PointCloud(data_support_points, colors=[0, 0, 0, 255])
scene.add_geometry(Qsupport)
scene.show()
