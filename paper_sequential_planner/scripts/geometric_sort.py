import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

xfake = np.linspace(0, 10, 100)
yfake = np.sin(xfake)
Xdata = np.vstack((xfake, yfake)).T
v = np.array([1, 0])


X = np.array(
    [
        [0.5, 1.5],
        [1.0, 0.5],
        [1.5, 1.0],
        [2.0, 2.5],
        [2.5, 1.0],
        [3.0, 3.5],
        [3.5, 3.0],
    ]
)

v = np.array([1, 1])
v = v / np.linalg.norm(v)
s = X @ v  # projection scalars
order = np.argsort(s)
X_sorted = X[order]

print("Original X:")
print(X)
print("Projection scalars:")
print(s)
print("Sorted order indices:")
print(order)
print("Sorted X:")
print(X_sorted)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], color="blue", label="Original Points")
ax.scatter(X_sorted[:, 0], X_sorted[:, 1], color="red", label="Sorted Points")

for i in range(X.shape[0]):
    ax.text(X[i, 0] + 0.1, X[i, 1], f"{i}", fontsize=12, color="blue")
    ax.text(X_sorted[i, 0], X_sorted[i, 1], f"{i}", fontsize=12, color="red")
ax.plot(
    [0, v[0] * 6],
    [0, v[1] * 6],
    color="green",
    linestyle="--",
    label="Projection Direction",
)
ax.legend()
ax.grid()
ax.set_aspect("equal", "box")
plt.show()
