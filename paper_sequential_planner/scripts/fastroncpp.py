import os
import numpy as np
import matplotlib.pyplot as plt
from fastronWrapper.fastronWrapper import PyFastron
from sklearn.metrics import accuracy_score
import time

np.random.seed(42)
np.set_printoptions(linewidth=1000, suppress=True, precision=2)
rsrc = os.environ["RSRC_DIR"]

# load dataset
dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
N_TRAIN = 9000
samples_id = np.random.choice(range(dataset.shape[0]), size=N_TRAIN, replace=False)
dataset_samples = dataset[samples_id]
data = dataset_samples[:, 0:2]
y = dataset_samples[:, 2]
data = np.ascontiguousarray(dataset_samples[:, :2])  # (N, 2), contiguous
y = np.ascontiguousarray(dataset_samples[:, 2:3])  # (N, 1), contiguous col vector

print(data.shape)  # (N, 2)
print(y.shape)  # (N, 1)

# FASTRON
# Initialize PyFastron
fastron = PyFastron(data)  # where data.shape = (N, d)
fastron.y = y  # where y.shape = (N,)
fastron.g = 10
fastron.maxUpdates = 5000
fastron.maxSupportPoints = 3000
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

# Predict values for a test set (ask for collision)
# data_test = np.array([[-3.14,0.2]])
# pred = fastron.eval(data_test) # where data_test.shape = (N_test, d)
# print(pred.shape)


# results
def ft_result():
    size = 360
    q1 = np.linspace(-np.pi, np.pi, size)
    q2 = np.linspace(-np.pi, np.pi, size)
    XX, YY = np.meshgrid(q1, q2)
    Q = np.column_stack([XX.ravel(), YY.ravel()])
    Y = fastron.eval(Q)
    ft_dataset = np.column_stack([Q, Y])
    return ft_dataset


alpha_trained = fastron.alpha
Gram = fastron.G
data_support_points = fastron.data
print(f"==>> data_support_points.shape: {data_support_points.shape}")

datafastron = ft_result()
gt_collision = dataset[dataset[:, 2] == 1][:, :2]
ft_collision = datafastron[datafastron[:, 2] == 1][:, :2]
train_point = dataset_samples[:, :2]

xtest = dataset[:, :2]
ytest = dataset[:, 2]
ypred = fastron.eval(xtest)
acc = accuracy_score(ytest, ypred)
print(f"Test accuracy: {acc*100:.2f}%")  # Test accuracy: 99.00%

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(gt_collision[:, 0], gt_collision[:, 1], "ro", markersize=3)
ax1.plot(
    train_point[:, 0],
    train_point[:, 1],
    "go",
    markersize=2,
    label="Training samples",
)
ax1.plot(data_support_points[:, 0], data_support_points[:, 1], "kx", markersize=5)
ax2.plot(ft_collision[:, 0], ft_collision[:, 1], "bo", markersize=3)
ax2.plot(data_support_points[:, 0], data_support_points[:, 1], "ko", markersize=2)
ax1.set_xlim(-np.pi, np.pi)
ax1.set_ylim(-np.pi, np.pi)
ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(-np.pi, np.pi)
ax1.set_aspect("equal")
ax2.set_aspect("equal")
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
plt.show()


class FastronCPP:

    def __init__(self):
        self.fastron = PyFastron(data)  # where data.shape = (N, d)
        self.fastron.y = y  # where y.shape = (N,)
        self.fastron.g = 10
        self.fastron.maxUpdates = 5000
        self.fastron.maxSupportPoints = 3000
        self.fastron.beta = 100

    def get_params(self):
        alpha_trained = fastron.alpha
        Gram = fastron.G
        data_support_points = fastron.data
        print(f"==>> data_support_points.shape: {data_support_points.shape}")

    def active_learning(self):
        start_time = time.time()
        self.fastron.activeLearning()
        end_time = time.time()
        print(f"Active Learning Time: {end_time - start_time:.2f} seconds")

    def update_model(self):
        start_time = time.time()
        self.fastron.updateModel()
        end_time = time.time()
        print(f"Model Update Time: {end_time - start_time:.2f} seconds")

    def update_models(self):
        pass

    def eval(self, Q):
        return self.fastron.eval(Q)

    def accuracy(self, xtest, ytest):
        ypred = self.fastron.eval(xtest)
        acc = accuracy_score(ytest, ypred)
        print(f"Test accuracy: {acc*100:.2f}%")
