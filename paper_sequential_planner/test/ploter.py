import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["svg.fonttype"] = "none"  # Render text as text, not paths
plt.rcParams["font.family"] = "serif"  # Use serif family for Computer Modern
plt.rcParams["font.serif"] = [
    "Computer Modern Roman",
    "CMU Serif",
    "Times",
    "DejaVu Serif",
]  # Computer Modern as first choice

# Configure math fonts separately
# plt.rcParams["mathtext.fontset"] = "stix"  # Options: 'dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom'
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern (LaTeX default)
# plt.rcParams["mathtext.fontset"] = "stixsans"  # STIX sans-serif math
# plt.rcParams["mathtext.fontset"] = "dejavuserif"  # DejaVu serif math

plt.rcParams["font.size"] = 10  # Base font size
plt.rcParams["axes.titlesize"] = 10  # Title font size
plt.rcParams["axes.labelsize"] = 10  # Axis label font size
plt.rcParams["xtick.labelsize"] = 10  # X-axis tick label font size
plt.rcParams["ytick.labelsize"] = 10  # Y-axis tick label font size
plt.rcParams["legend.fontsize"] = 10  # Legend font size
plt.rcParams["figure.titlesize"] = 10  # Figure title font size

# print([f.name for f in font_manager.fontManager.ttflist])
# plotting
fig, axes = plt.subplots(2, 1, figsize=(3.4861, 0.8 * 3.4861), sharex=True)

times = np.linspace(0, 1, 100)
y1 = 2.2 * np.sin(2 * np.pi * times)
y2 = 2.2 * np.cos(2 * np.pi * times)
# plot normal ----------------
axes[0].plot(
    times,
    y1,
    color="gray",
    linewidth=3,
    linestyle="--",
    label="unwrapped path",
)
axes[1].plot(
    times,
    y2,
    color="gray",
    linewidth=3,
    linestyle="--",
    label="unwrapped path",
)

axes[0].fill_between([0.0, 1.0], np.pi, 2 * np.pi, color="blue", alpha=0.2)
axes[0].fill_between([0.0, 1.0], -np.pi, np.pi, color="green", alpha=0.2)
axes[0].fill_between([0.0, 1.0], -2 * np.pi, -np.pi, color="blue", alpha=0.2)
axes[1].fill_between([0.0, 1.0], np.pi, 2 * np.pi, color="blue", alpha=0.2)
axes[1].fill_between([0.0, 1.0], -np.pi, np.pi, color="green", alpha=0.2)
axes[1].fill_between([0.0, 1.0], -2 * np.pi, -np.pi, color="blue", alpha=0.2)

axes[0].set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
axes[0].set_yticklabels(
    ["$-2\\pi$", "$-\\pi$", "$\\theta_1$", "$\\pi$", "$2\\pi$"]
)
axes[1].set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
axes[1].set_yticklabels(
    ["$-2\\pi$", "$-\\pi$", "$\\theta_2$", "$\\pi$", "$2\\pi$"]
)

axes[0].set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])
axes[0].set_xticklabels(["0", "0.2", "0.4", "$t$", "0.6", "0.8", "1"])
axes[1].set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])
axes[1].set_xticklabels(["0", "0.2", "0.4", "$t$", "0.6", "0.8", "1"])

axes[1].set_xlim((0.0, 1.0))
axes[0].set_ylim((-2 * np.pi, 2 * np.pi))
axes[1].set_ylim((-2 * np.pi, 2 * np.pi))

axes[0].legend(loc="upper left")
axes[1].legend(loc="upper left")

axes[1].set_title("Wrapped Joint $\\mathcal{C}_{obs}$ over Time")
fig.tight_layout(pad=0)
fig.savefig("/home/demo_plot.svg")

plt.rcParams["savefig.dpi"] = 300  # High DPI for rasterized elements
fig, ax = plt.subplots(figsize=(8, 6))
collision_points = np.random.uniform(-5, 5, (10000, 2))
num_edges = 1000
for i in range(num_edges):
    start = np.random.uniform(-5, 5, 2)
    end = start + np.random.normal(0, 0.5, 2)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        "gray",
        alpha=0.3,
        linewidth=0.5,
        rasterized=True,
    )
# Dense collision points (rasterized)
ax.scatter(
    collision_points[:, 0],
    collision_points[:, 1],
    s=1,
    color="darkcyan",
    alpha=0.7,
    rasterized=True,
)
# Important path points (vectors - stay crisp)
path_points = np.array([[-4, -4], [-2, 0], [0, 2], [2, 0], [4, 4]])
ax.plot(
    path_points[:, 0],
    path_points[:, 1],
    "blue",
    linewidth=3,
    marker="o",
    markersize=8,
    markerfacecolor="yellow",
    markeredgecolor="black",
)
# Start/goal points (vectors - important to stay crisp)
ax.scatter(
    [-4],
    [-4],
    s=200,
    color="green",
    edgecolor="black",
    linewidth=2,
    marker="s",
    zorder=20,
)
ax.scatter(
    [4],
    [4],
    s=200,
    color="red",
    edgecolor="black",
    linewidth=2,
    marker="s",
    zorder=20,
)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.grid(True, alpha=0.3)
ax.set_title("Optimized Plot for Your Use Case")
plt.savefig(
    "optimized_robotics_plot.svg",
    format="svg",
    bbox_inches="tight",
    dpi=300,
)
