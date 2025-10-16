import numpy as np
from spatial_geometry.utils import Utils
import pandas as pd
from eaik.IK_DH import DhRobot
import os
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=2000)
rsrcpath = os.environ["RSRC_DIR"] + "/rnd_torus"

limt6 = np.array(
    [
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-np.pi, np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
    ]
)


def ur5e_dh():
    d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
    alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
    a = np.array([0, -0.425, -0.3922, 0, 0, 0])
    bot = DhRobot(alpha, a, d)
    return bot


def solve_fk(bot, angles):
    return bot.fwdKin(angles)


def solve_ik(bot, h):
    sols = bot.IK(h)
    return sols.num_solutions(), sols.Q


# inside pi
qs = np.array(
    (
        0.06613874435424827,
        -1.1243596076965368,
        1.8518862724304164,
        -0.8598046302795428,
        1.7857480049133336,
        0.1984162330627437,
    )
)
# outside pi
qs = np.array(
    (
        0.06613874435424827 - 2 * np.pi,
        -1.1243596076965368 + 2 * np.pi,
        1.8518862724304164 - 2 * np.pi,
        -0.8598046302795428 + 2 * np.pi,
        1.7857480049133336 - 2 * np.pi,
        0.1984162330627437 - 2 * np.pi,
    )
)
bot = ur5e_dh()
q = np.array(
    (
        -2.976245641708381,
        -1.0582203865051305,
        1.587330818176266,
        -0.7936654090881332,
        1.653469562530514,
        0.13227748870849654,
    )
)

h = solve_fk(bot, q)
ns, Qik = solve_ik(bot, h)

dist = np.linalg.norm(Qik - qs, axis=1)
print("dist=", dist)
print("min dist=", np.min(dist), " at index ", np.argmin(dist))

# Create bar plot for d values
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
x_positions = np.arange(dist.shape[0])
ax.bar(x_positions, dist, width=0.8, label="Euclidean", alpha=0.7)
ax.set_xlabel("Index")
ax.set_ylabel("Norm Difference")
ax.set_title("Bar Graph of Norm Differences")
ax.set_xticks(x_positions)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

QaikAlt1 = Utils.find_alt_config(Qik[0].reshape(6, 1), limt6).T
QaikAlt2 = Utils.find_alt_config(Qik[1].reshape(6, 1), limt6).T
QaikAlt3 = Utils.find_alt_config(Qik[2].reshape(6, 1), limt6).T
QaikAlt4 = Utils.find_alt_config(Qik[3].reshape(6, 1), limt6).T
QaikAlt5 = Utils.find_alt_config(Qik[4].reshape(6, 1), limt6).T
QaikAlt6 = Utils.find_alt_config(Qik[5].reshape(6, 1), limt6).T
QaikAlt7 = Utils.find_alt_config(Qik[6].reshape(6, 1), limt6).T
QaikAlt8 = Utils.find_alt_config(Qik[7].reshape(6, 1), limt6).T

Deul1 = np.linalg.norm(QaikAlt1 - qs, axis=1)
Deul2 = np.linalg.norm(QaikAlt2 - qs, axis=1)
Deul3 = np.linalg.norm(QaikAlt3 - qs, axis=1)
Deul4 = np.linalg.norm(QaikAlt4 - qs, axis=1)
Deul5 = np.linalg.norm(QaikAlt5 - qs, axis=1)
Deul6 = np.linalg.norm(QaikAlt6 - qs, axis=1)
Deul7 = np.linalg.norm(QaikAlt7 - qs, axis=1)
Deul8 = np.linalg.norm(QaikAlt8 - qs, axis=1)

min1 = np.min(Deul1)
min2 = np.min(Deul2)
min3 = np.min(Deul3)
min4 = np.min(Deul4)
min5 = np.min(Deul5)
min6 = np.min(Deul6)
min7 = np.min(Deul7)
min8 = np.min(Deul8)

minall = np.min(
    [min1, min2, min3, min4, min5, min6, min7, min8]
)  # minimum distance among all alternative configurations
print("min1=", min1)
print("min2=", min2)
print("min3=", min3)
print("min4=", min4)
print("min5=", min5)
print("min6=", min6)
print("min7=", min7)
print("min8=", min8)
print("minall=", minall)


# Collect all distance data for 8 IK branches (each has 32 alt configs)
dist_data = [Deul1, Deul2, Deul3, Deul4, Deul5, Deul6, Deul7, Deul8]

# Create grouped bar plot
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

ax.axhline(y=minall, color="r", linestyle="--", label="Minimum Distance")

# Parameters for grouping
n_branches = 8
n_configs = 32
branch_ids = np.arange(1, n_branches + 1)
group_width = 0.8  # Total width for each group
bar_width = group_width / n_configs  # Width of individual bars
group_spacing = 1.0  # Spacing between groups

# Plot bars for each IK branch
colors = plt.cm.Set3(np.linspace(0, 1, n_branches))
for branch_idx, (branch_id, distances) in enumerate(zip(branch_ids, dist_data)):
    # Calculate x positions for this branch's bars
    branch_center = branch_id * group_spacing
    x_positions = branch_center + np.linspace(
        -group_width / 2, group_width / 2, n_configs
    )

    # Plot bars for this branch
    bars = ax.bar(
        x_positions,
        distances,
        width=bar_width,
        alpha=0.7,
        color=colors[branch_idx],
        label=f"IK Branch {branch_id}",
    )

# Customize the plot
ax.set_xlabel("IK Branch ID")
ax.set_ylabel("Euclidean Distance")
ax.set_title(
    "Alternative Configuration Distances by IK Branch\n(32 configurations per branch)"
)

# Set x-axis ticks at branch centers
branch_centers = branch_ids * group_spacing
ax.set_xticks(branch_centers)
ax.set_xticklabels(branch_ids)

# Add grid and legend
ax.grid(True, alpha=0.3, axis="y")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()


raise Exception("stop")

alt = Utils.find_alt_config(q.reshape(-1, 1), limt6).T
bin = np.linspace(0, 8, 8)

diff = alt - qs
print("alt=\n", alt)
print("diff alt=\n", diff)

d = np.linalg.norm(diff, axis=1)
print("norm diff alt=", d)
print("d shape:", d.shape)

dtorus = []
for i in range(alt.shape[0]):
    dtorus.append(
        Utils.minimum_dist_torus(qs.reshape(6, 1), alt[i, :].reshape(6, 1))
    )
print("dtorus=", dtorus)


# Create bar plot for d values
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
x_positions = np.arange(len(d))  # Integer positions from 0 to 31
ax.bar(x_positions, d, width=0.8, label="Euclidean", alpha=0.7)
ax.bar(x_positions, dtorus, width=0.4, alpha=0.7, label="Torus")
ax.set_xlabel("Index")
ax.set_ylabel("Norm Difference")
ax.set_title("Bar Graph of Norm Differences")
ax.set_xticks(x_positions)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
