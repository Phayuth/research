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
fig.tight_layout(pad=0)
fig.savefig("/home/ /demo_plot.svg")