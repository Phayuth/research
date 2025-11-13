import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"  # Render text as text, not paths


def retimer_tour(path_tour):
    _, first_occ_ind = np.unique(path_tour, axis=0, return_index=True)
    sorted_indices = np.sort(first_occ_ind)
    path_tour_nodup = path_tour[sorted_indices]

    path_tour_interp = []
    for i in range(path_tour_nodup.shape[0] - 1):
        pi_nm = np.linspace(path_tour_nodup[i], path_tour_nodup[i + 1], 10)
        for px in pi_nm:
            path_tour_interp.append(px)
    path_tour_interp = np.vstack(path_tour_interp)
    return path_tour_interp


def compute_cf_cost():
    tour_normal = np.load("topaper_/cf_tour_normal_1.npy", allow_pickle=True)
    tour_altconfig = np.load("topaper_/cf_tour_altconfig_1.npy", allow_pickle=True)
    path_nm = retimer_tour(tour_normal)
    path_ac = retimer_tour(tour_altconfig)

    cost = 0.0
    for i in range(1, path_nm.shape[0]):
        diff = path_nm[i] - path_nm[i - 1]
        cost += np.linalg.norm(diff)
    print("Normal config cost:", cost)

    costac = 0.0
    for i in range(1, path_ac.shape[0]):
        diff = path_ac[i] - path_ac[i - 1]
        costac += np.linalg.norm(diff)
    print("Alt config cost:", costac)

def fig_joint_time_():

    tour_normal = np.load("./cf_tour_normal_2.npy", allow_pickle=True)
    tour_altconfig = np.load("./cf_tour_altconfig_2.npy", allow_pickle=True)
    path_nm = retimer_tour(tour_normal)
    path_ac = retimer_tour(tour_altconfig)

    times_nm = np.linspace(0.0, 1.0, path_nm.shape[0])
    times_ac = np.linspace(0.0, 1.0, path_ac.shape[0])

    # plotting
    fig, axes = plt.subplots(6, 1, figsize=(3.4861, 1.3 * 3.4861), sharex=True)

    for i in range(6):
        axes[i].plot(
            times_nm,
            path_nm[:, i],
            color="gray",
            linewidth=3,
            linestyle="-",
            label="Normal config",
        )
        axes[i].plot(
            times_ac,
            path_ac[:, i],
            color="orange",
            linewidth=3,
            linestyle="-",
            label="Alt config",
        )

    for i in range(6):
        axes[i].fill_between([0.0, 1.0], np.pi, 2 * np.pi, color="blue", alpha=0.2)
        axes[i].fill_between([0.0, 1.0], -np.pi, np.pi, color="green", alpha=0.2)
        axes[i].fill_between(
            [0.0, 1.0], -2 * np.pi, -np.pi, color="blue", alpha=0.2
        )

        # axes[i].set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
        # axes[i].set_yticklabels(
        #     [
        #         "$-2\\pi$",
        #         "$-\\pi$",
        #         f"$\\theta_{{{i+1}}}$",
        #         "$\\pi$",
        #         "$2\\pi$",
        #     ]
        # )
        axes[i].set_yticks([0])
        axes[i].set_yticklabels([f"$\\theta_{{{i+1}}}$"])
        axes[i].set_xlim((0.0, 1.0))
        axes[i].set_ylim((-2 * np.pi, 2 * np.pi))

    axes[-1].set_xticks(np.linspace(0.0, 1.0, 26))
    tname = [f"{i}" for i in range(1, 25)]
    tname.insert(0, "Init")
    tname.append("Init")
    axes[-1].set_xticklabels(tname, rotation=90)
    fig.tight_layout(pad=0)

    plt.savefig("/home/ /qs2_.svg", format="svg", bbox_inches="tight")
    plt.show()


def generate_fig_label():
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot(
        [0, 1],
        [0, 1],
        color="orange",
        linewidth=3,
        linestyle="-",
        label="Alt config",
    )
    ax.plot(
        [0, 1],
        [1, 0],
        color="gray",
        linewidth=3,
        linestyle="-",
        label="Normal config",
    )
    ax.legend(loc="center")
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.savefig("/home/ /fig_label.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    fig_joint_time_()
    # generate_fig_label()
    # compute_cf_cost()
