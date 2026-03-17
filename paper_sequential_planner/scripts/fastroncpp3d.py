import os
import numpy as np
import matplotlib.pyplot as plt
from fastronWrapper.fastronWrapper import PyFastron
from sklearn.metrics import accuracy_score
import time
import trimesh


np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]

dataset = np.load(os.path.join(rsrc, "spatial3r_cspace.npy"))
dataset = dataset.copy()
dataset[:, 3] = np.where(dataset[:, 3] == 0, -1, dataset[:, 3])
N_TRAIN = 10000
samples_id = np.random.choice(range(dataset.shape[0]), size=N_TRAIN, replace=False)
dataset_samples = dataset[samples_id]
data = dataset_samples[:, 0:3]
y = dataset_samples[:, 3]
data = np.ascontiguousarray(dataset_samples[:, :3])  # (N, 3), contiguous
y = np.ascontiguousarray(dataset_samples[:, 3:4])  # (N, 1), contiguous col vector


print(data.shape)  # (N, 3)
print(y.shape)  # (N, 1)

# FASTRON
# Initialize PyFastron
fastron = PyFastron(data)  # where data.shape = (N, d)
fastron.y = y  # where y.shape = (N,)
fastron.g = 10
fastron.maxUpdates = 5000
fastron.maxSupportPoints = 4000
fastron.beta = 100


# Active Learning
start_time = time.time()
fastron.activeLearning()
end_time = time.time()
print(f"Active Learning Time: {end_time - start_time:.2f} seconds")

start_time = time.time()
# Update label
# fastron.updateLabels()
# Train model
fastron.updateModel()
end_time = time.time()
print(f"Model Update Time: {end_time - start_time:.2f} seconds")


# results
def ft_result():
    size = 360
    q1 = np.linspace(-np.pi, np.pi, size)
    q2 = np.linspace(-np.pi, np.pi, size)
    q3 = np.linspace(-np.pi, np.pi, size)
    XX, YY, ZZ = np.meshgrid(q1, q2, q3)
    Q = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    Y = fastron.eval(Q)
    ft_dataset = np.column_stack([Q, Y])
    return ft_dataset


alpha_trained = fastron.alpha
Gram = fastron.G
data_support_points = fastron.data
print(f"==>> data_support_points.shape: {data_support_points.shape}")

datafastron = ft_result()
gt_collision = dataset[dataset[:, 3] == 1][:, :3]
ft_collision = datafastron[datafastron[:, 3] == 1][:, :3]
train_point = dataset_samples[:, :3]

xtest = dataset[:, :3]
ytest = dataset[:, 3]
ypred = fastron.eval(xtest)
acc = accuracy_score(ytest, ypred)
print(f"Test accuracy: {acc*100:.2f}%")  # Test accuracy: 88.30%

sc = trimesh.Scene()
Qfree = dataset[dataset[:, -1] == 0][:, :3]
Qcoll = dataset[dataset[:, -1] == 1][:, :3]
axis = trimesh.creation.axis(origin_size=0.05, axis_length=np.pi)
box = trimesh.creation.box(extents=(2 * np.pi, 2 * np.pi, 2 * np.pi))
box.visual.face_colors = [100, 150, 255, 40]
scene = trimesh.Scene()
scene.add_geometry(box)
scene.add_geometry(axis)
# Qr = trimesh.points.PointCloud(Qcoll, colors=[255, 0, 0, 255])
# scene.add_geometry(Qr)
Qfastron = trimesh.points.PointCloud(ft_collision, colors=[0, 0, 255, 255])
scene.add_geometry(Qfastron)
# Qtrain = trimesh.points.PointCloud(train_point, colors=[0, 255, 0, 255])
# scene.add_geometry(Qtrain)
Qsupport = trimesh.points.PointCloud(data_support_points, colors=[0, 0, 0, 255])
scene.add_geometry(Qsupport)
scene.show()
