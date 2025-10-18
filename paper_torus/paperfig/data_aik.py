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
qs_inpi = np.array(
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
qs_outpi = np.array(
    (
        0.06613874435424827 - 2 * np.pi,
        -1.1243596076965368 + 2 * np.pi,
        1.8518862724304164 - 2 * np.pi,
        -0.8598046302795428 + 2 * np.pi,
        1.7857480049133336 - 2 * np.pi,
        0.1984162330627437 - 2 * np.pi,
    )
)

# qs_outpi = np.random.uniform(-np.pi, np.pi, size=(6,))
# for i in range(6):
#     if qs_outpi[i] <= np.pi:
#         qs_outpi[i] += 2 * np.pi
#     else:
#         qs_outpi[i] -= 2 * np.pi


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


def distance_on_euclidean_branches(qstart, Qendbranches):
    diff = Qendbranches - qstart
    dists = np.linalg.norm(diff, axis=1)
    min_dist = np.min(dists)
    min_index = np.argmin(dists)
    return dists, min_dist, min_index


def distance_on_torus_branches(qstart, Qendbranches):
    dists = []
    for i in range(Qendbranches.shape[0]):
        dist = Utils.minimum_dist_torus(
            qstart.reshape(-1, 1), Qendbranches[i, :].reshape(-1, 1)
        )
        dists.append(dist)
    dists = np.array(dists)
    min_dist = np.min(dists)
    min_index = np.argmin(dists)
    return dists, min_dist, min_index


def distance_on_euclidean_altconfig(qstart, qend):
    Qendalt = Utils.find_alt_config(qend.reshape(-1, 1), limt6).T
    diff = Qendalt - qstart
    dists = np.linalg.norm(diff, axis=1)
    min_dist = np.min(dists)
    min_index = np.argmin(dists)
    return dists, min_dist, min_index


def compare_euclidean_branches():
    # the distance qs in pi with always smaller than qs out pi because solution
    # of analytical IK are always all in -pi to pi
    disteul_inpi, mindist_eul_inpi, minidx_eul_inpi = (
        distance_on_euclidean_branches(qs_inpi, Qik)
    )
    disteul_outpi, mindist_eul_outpi, minidx_eul_outpi = (
        distance_on_euclidean_branches(qs_outpi, Qik)
    )
    disttor_inpi, mindist_tor_inpi, minidx_tor_inpi = distance_on_torus_branches(
        qs_inpi, Qik
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    y_positions = np.arange(disteul_inpi.shape[0])
    ax.barh(
        y_positions, disteul_inpi, height=0.8, label="Euclidean In PI", alpha=0.7
    )
    ax.barh(
        y_positions, disteul_outpi, height=0.4, label="Euclidean Out PI", alpha=0.7
    )
    ax.barh(y_positions, disttor_inpi, height=0.4, label="Torus In PI", alpha=0.7)
    ax.set_ylabel("Index")
    ax.set_xlabel("Norm Difference")
    ax.set_title("Horizontal Bar Graph of Norm Differences")
    ax.set_yticks(y_positions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_euclidean_alt():
    qaik = Qik[3]
    disteul_alt_inpi, mindist_eul_alt_inpi, minidx_eul_alt_inpi = (
        distance_on_euclidean_altconfig(qs_inpi, qaik)
    )
    disteul_alt_outpi, mindist_eul_alt_outpi, minidx_eul_alt_outpi = (
        distance_on_euclidean_altconfig(qs_outpi, qaik)
    )
    disttor = Utils.minimum_dist_torus(qs_inpi.reshape(-1, 1), qaik.reshape(-1, 1))

    distog = np.linalg.norm(qaik - qs_inpi)
    print("Distance on original configuration:", distog)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    y_positions = np.arange(disteul_alt_inpi.shape[0])
    ax.barh(
        y_positions,
        disteul_alt_inpi,
        height=0.8,
        label="Euclidean In PI",
        alpha=0.7,
    )
    ax.barh(
        y_positions,
        disteul_alt_outpi,
        height=0.4,
        label="Euclidean Out PI",
        alpha=0.7,
    )
    ax.barh(y_positions, disttor, height=0.4, label="Torus", alpha=0.7)
    ax.set_ylabel("Index")
    ax.set_xlabel("Norm Difference")
    ax.set_title("Norm Differences (Alt Configs) of 1 of 8 IK Branches")
    ax.set_yticks(y_positions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_branches_altconfig():
    Deul1, min_dist1, min_index1 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[0, :]
    )
    Deul2, min_dist2, min_index2 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[1, :]
    )
    Deul3, min_dist3, min_index3 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[2, :]
    )
    Deul4, min_dist4, min_index4 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[3, :]
    )
    Deul5, min_dist5, min_index5 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[4, :]
    )
    Deul6, min_dist6, min_index6 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[5, :]
    )
    Deul7, min_dist7, min_index7 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[6, :]
    )
    Deul8, min_dist8, min_index8 = distance_on_euclidean_altconfig(
        qs_inpi, Qik[7, :]
    )
    dist_data = [Deul1, Deul2, Deul3, Deul4, Deul5, Deul6, Deul7, Deul8]
    mindata_per = np.array(
        [
            min_dist1,
            min_dist2,
            min_dist3,
            min_dist4,
            min_dist5,
            min_dist6,
            min_dist7,
            min_dist8,
        ]
    )

    Dtor1 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[0, :].reshape(-1, 1)
    )
    Dtor2 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[1, :].reshape(-1, 1)
    )
    Dtor3 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[2, :].reshape(-1, 1)
    )
    Dtor4 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[3, :].reshape(-1, 1)
    )
    Dtor5 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[4, :].reshape(-1, 1)
    )
    Dtor6 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[5, :].reshape(-1, 1)
    )
    Dtor7 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[6, :].reshape(-1, 1)
    )
    Dtor8 = Utils.minimum_dist_torus(
        qs_inpi.reshape(-1, 1), Qik[7, :].reshape(-1, 1)
    )
    dist_tor_data = [Dtor1, Dtor2, Dtor3, Dtor4, Dtor5, Dtor6, Dtor7, Dtor8]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Parameters for grouping
    n_branches = 8
    n_configs = 32
    branch_ids = np.arange(1, n_branches + 1)
    group_height = 0.8  # Total height for each group
    bar_height = group_height / n_configs  # Height of individual bars
    group_spacing = 1.0  # Spacing between groups

    # Plot horizontal bars for each IK branch
    colors = plt.cm.Set3(np.linspace(0, 1, n_branches))
    for branch_idx, (branch_id, distances, torus_distances) in enumerate(
        zip(branch_ids, dist_data, dist_tor_data)
    ):
        # Calculate y positions for this branch's bars
        branch_center = branch_id * group_spacing
        y_positions = branch_center + np.linspace(
            -group_height / 2, group_height / 2, n_configs
        )

        # Plot horizontal bars for this branch
        bars = ax.barh(
            y_positions,
            distances,
            height=bar_height,
            alpha=1.0,
            color=colors[branch_idx],
            label=f"IK Branch {branch_id}",
        )
        ax.barh(
            y_positions,
            torus_distances,
            height=bar_height,
            alpha=0.5,
            # color=colors[branch_idx],
        )
    mineul = np.min(mindata_per)
    mintor = np.min(mindata_per)
    print("Minimum Euclidean Distance among all branches:", mineul)
    print("Minimum Torus Distance among all branches:", mintor)
    ax.axvline(
        x=np.min(mindata_per),
        color="r",
        linestyle="--",
        label="Minimum Euclidean Distance",
        linewidth=2,
    )

    ax.axvline(
        x=np.min(mindata_per),
        color="b",
        linestyle="--",
        label="Minimum Torus Distance",
    )
    ax.set_ylabel("IK Branch ID")
    ax.set_xlabel("Distance")
    branch_centers = branch_ids * group_spacing
    ax.set_yticks(branch_centers)
    ax.set_yticklabels(branch_ids)
    ax.grid(True, alpha=0.3, axis="x")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def compare_numerical():
    de_inpi, mine_inpi, minide_inpi = distance_on_euclidean_branches(qs_inpi, Qik)
    de_alt1_inpi, mde_alt1_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[0, :]
    )
    de_alt2_inpi, mde_alt2_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[1, :]
    )
    de_alt3_inpi, mde_alt3_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[2, :]
    )
    de_alt4_inpi, mde_alt4_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[3, :]
    )
    de_alt5_inpi, mde_alt5_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[4, :]
    )
    de_alt6_inpi, mde_alt6_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[5, :]
    )
    de_alt7_inpi, mde_alt7_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[6, :]
    )
    de_alt8_inpi, mde_alt8_inpi, _ = distance_on_euclidean_altconfig(
        qs_inpi, Qik[7, :]
    )
    print("Distance qs_inpi to 8 branches:", de_inpi)
    print("Distance qs_inpi to alt of branch 1:", de_alt1_inpi)
    print("Distance qs_inpi to alt of branch 2:", de_alt2_inpi)
    print("Distance qs_inpi to alt of branch 3:", de_alt3_inpi)
    print("Distance qs_inpi to alt of branch 4:", de_alt4_inpi)
    print("Distance qs_inpi to alt of branch 5:", de_alt5_inpi)
    print("Distance qs_inpi to alt of branch 6:", de_alt6_inpi)
    print("Distance qs_inpi to alt of branch 7:", de_alt7_inpi)
    print("Distance qs_inpi to alt of branch 8:", de_alt8_inpi)
    # dist_data = [Deul1, Deul2, Deul3, Deul4, Deul5, Deul6, Deul7, Deul8]
    mindata_per = np.array(
        [
            mde_alt1_inpi,
            mde_alt2_inpi,
            mde_alt3_inpi,
            mde_alt4_inpi,
            mde_alt5_inpi,
            mde_alt6_inpi,
            mde_alt7_inpi,
            mde_alt8_inpi,
        ]
    )
    minofalt_inpi = np.min(mindata_per)
    print("Minimum of alt configs in pi:", minofalt_inpi)

    print("----- Outside PI -----")
    de_outpi, mine_outpi, minide_outpi = distance_on_euclidean_branches(
        qs_outpi, Qik
    )
    de_alt1_outpi, mde_alt1_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[0, :]
    )
    de_alt2_outpi, mde_alt2_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[1, :]
    )
    de_alt3_outpi, mde_alt3_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[2, :]
    )
    de_alt4_outpi, mde_alt4_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[3, :]
    )
    de_alt5_outpi, mde_alt5_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[4, :]
    )
    de_alt6_outpi, mde_alt6_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[5, :]
    )
    de_alt7_outpi, mde_alt7_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[6, :]
    )
    de_alt8_outpi, mde_alt8_outpi, _ = distance_on_euclidean_altconfig(
        qs_outpi, Qik[7, :]
    )

    print("Distance qs_outpi to 8 branches:", de_outpi)
    print("Distance qs_outpi to alt of branch 1:", de_alt1_outpi)
    print("Distance qs_outpi to alt of branch 2:", de_alt2_outpi)
    print("Distance qs_outpi to alt of branch 3:", de_alt3_outpi)
    print("Distance qs_outpi to alt of branch 4:", de_alt4_outpi)
    print("Distance qs_outpi to alt of branch 5:", de_alt5_outpi)
    print("Distance qs_outpi to alt of branch 6:", de_alt6_outpi)
    print("Distance qs_outpi to alt of branch 7:", de_alt7_outpi)
    print("Distance qs_outpi to alt of branch 8:", de_alt8_outpi)
    mindata_per_outpi = np.array(
        [
            mde_alt1_outpi,
            mde_alt2_outpi,
            mde_alt3_outpi,
            mde_alt4_outpi,
            mde_alt5_outpi,
            mde_alt6_outpi,
            mde_alt7_outpi,
            mde_alt8_outpi,
        ]
    )
    minofalt_outpi = np.min(mindata_per_outpi)
    print("Minimum of alt configs out pi:", minofalt_outpi)


def compare_qs_to_256_altconfigs(qs):
    Deul, minde, minidxe = distance_on_euclidean_branches(qs, Qik)
    de_alt1, mde_alt1, _ = distance_on_euclidean_altconfig(qs, Qik[0, :])
    de_alt2, mde_alt2, _ = distance_on_euclidean_altconfig(qs, Qik[1, :])
    de_alt3, mde_alt3, _ = distance_on_euclidean_altconfig(qs, Qik[2, :])
    de_alt4, mde_alt4, _ = distance_on_euclidean_altconfig(qs, Qik[3, :])
    de_alt5, mde_alt5, _ = distance_on_euclidean_altconfig(qs, Qik[4, :])
    de_alt6, mde_alt6, _ = distance_on_euclidean_altconfig(qs, Qik[5, :])
    de_alt7, mde_alt7, _ = distance_on_euclidean_altconfig(qs, Qik[6, :])
    de_alt8, mde_alt8, _ = distance_on_euclidean_altconfig(qs, Qik[7, :])
    de_altall = np.stack(
        [
            de_alt1,
            de_alt2,
            de_alt3,
            de_alt4,
            de_alt5,
            de_alt6,
            de_alt7,
            de_alt8,
        ]
    )
    de_altall = de_altall.flatten()
    mindist_altall = np.min(de_altall)

    print("------QS:", qs, "------")
    print("Distance from qs to 8 branches:", Deul)
    print("and min distance:", minde, " at index ", minidxe)
    print("There are {} alt configs in total.".format(de_altall.shape[0]))
    # print("Distance from qs to all alt configs of 8 branches:", de_altall)
    print("Minimum distance among all alt configs:", mindist_altall)


if __name__ == "__main__":
    # compare_euclidean_branches()
    # compare_euclidean_alt()
    # compare_branches_altconfig()
    # compare_numerical()

    # qs inside pi
    compare_qs_to_256_altconfigs(qs_inpi)

    # qs outside pi
    compare_qs_to_256_altconfigs(qs_outpi)
