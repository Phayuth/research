import numpy as np
from spatial_geometry.utils import Utils
import pandas as pd

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

limt6ontable = np.array(
    [
        [-2 * np.pi, 2 * np.pi],
        [-np.pi, 0],
        [-np.pi, np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
    ]
)

qs = np.array(
    [
        0.39683294296264565,
        -0.9920818805694598,
        1.5873310565948522,
        -0.8928737640380842,
        1.8518865108490026,
        -3.141592741012566,
    ]
)
qe = np.array(
    [
        -3.6376335620880056,
        -1.7857475280761683,
        -1.7196087837219274,
        -2.1825799942016664,
        -1.9180250167846644,
        -0.2645549774169931,
    ]
)
qalt = Utils.find_alt_config(qe.reshape(-1, 1), limt6, filterOriginalq=False).T


def individual_joint_move_on_euclidean(qs, qe):
    diff = np.abs(qe - qs)
    return diff


def individual_joint_move_on_torus(qa, qb):
    qa = qa.reshape(-1, 1)
    qb = qb.reshape(-1, 1)
    L = np.full_like(qa, 2 * np.pi)
    delta = np.abs(qa - qb)
    deltaw = L - delta
    deltat = np.min(np.hstack((delta, deltaw)), axis=1)
    return np.abs(deltat)


jointmove_eul = np.zeros_like(qalt)
jointmove_tor = np.zeros_like(qalt)
for i in range(qalt.shape[0]):
    jointmove_eul[i, :] = individual_joint_move_on_euclidean(qs, qalt[i, :])
    jointmove_tor[i, :] = individual_joint_move_on_torus(qs, qalt[i, :])


# on torus, the minimum distance will always be the same as original
distance_eul = np.linalg.norm(jointmove_eul, axis=1)
distance_tor = np.linalg.norm(jointmove_tor, axis=1)

# write Qalt to csv
df = pd.DataFrame(qalt)
df.to_csv("data_nik_qalt.csv", index=False)

# write qgoalalt, jointmove_eul, jointmove_tor, distance_eul, distance_tor to csv
df_dist = pd.DataFrame(
    {
        "qgoalalt_0": qalt[:, 0],
        "qgoalalt_1": qalt[:, 1],
        "qgoalalt_2": qalt[:, 2],
        "qgoalalt_3": qalt[:, 3],
        "qgoalalt_4": qalt[:, 4],
        "qgoalalt_5": qalt[:, 5],
        "jointmove_eul_0": jointmove_eul[:, 0],
        "jointmove_eul_1": jointmove_eul[:, 1],
        "jointmove_eul_2": jointmove_eul[:, 2],
        "jointmove_eul_3": jointmove_eul[:, 3],
        "jointmove_eul_4": jointmove_eul[:, 4],
        "jointmove_eul_5": jointmove_eul[:, 5],
        "jointmove_tor_0": jointmove_tor[:, 0],
        "jointmove_tor_1": jointmove_tor[:, 1],
        "jointmove_tor_2": jointmove_tor[:, 2],
        "jointmove_tor_3": jointmove_tor[:, 3],
        "jointmove_tor_4": jointmove_tor[:, 4],
        "jointmove_tor_5": jointmove_tor[:, 5],
        "distance_eul": distance_eul,
        "distance_tor": distance_tor,
    }
)
df_dist.to_csv("data_nik_dist.csv", index=False)


def individual_joint_move_on_euclidean_collisionfree_path(id):
    df_path = pd.read_csv(f"./paths/data_euclidean_path_goal_{id}.csv")
    path = df_path.to_numpy()
    pathdiff = np.diff(path, axis=0)
    pdabs = np.abs(pathdiff)
    joint_moves = np.sum(pdabs, axis=0)
    return joint_moves


def compute_joint_moves_on_euclidean_collisionfree_path():
    joint_moves_all = []
    for i in range(32):
        joint_moves = individual_joint_move_on_euclidean_collisionfree_path(i)
        joint_moves_all.append(joint_moves)
    joint_moves_all = np.array(joint_moves_all)
    df_jm = pd.DataFrame(
        joint_moves_all, columns=[f"jointmove_eul_{j}" for j in range(6)]
    )
    df_jm.to_csv(
        "data_nik_euclidean_collisionfree_path_jointmoves.csv", index=False
    )


def distance_on_euclidean_altconfig(qstart, qend):
    Qendalt = Utils.find_alt_config(qend.reshape(-1, 1), limt6).T
    diff = Qendalt - qstart
    dists = np.linalg.norm(diff, axis=1)
    min_dist = np.min(dists)
    min_index = np.argmin(dists)
    return dists, min_dist, min_index


def compare_data():
    import matplotlib.pyplot as plt
    import pandas as pd

    disteul, min_dist_eul, min_index_eul = distance_on_euclidean_altconfig(qs, qe)
    disttor = Utils.minimum_dist_torus(qs.reshape(-1, 1), qe.reshape(-1, 1))
    distcolfree = pd.read_csv("data_planner_results.csv")["cost"].values

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    y_positions = np.arange(disteul.shape[0])
    ax.barh(y_positions, disteul, height=0.8, label="Euclidean", alpha=0.7)
    ax.barh(y_positions, disttor, height=0.4, label="Torus", alpha=0.7)
    ax.barh(
        y_positions,
        distcolfree,
        height=0.2,
        label="Collision Free",
        alpha=0.7,
        color="r",
    )
    ax.set_ylabel("Index")
    ax.set_xlabel("Norm Difference")
    ax.set_title("Horizontal Bar Graph of Norm Differences")
    ax.set_yticks(y_positions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # compute_joint_moves_on_euclidean_collisionfree_path()
    # compare_data()

    qaltnew = Utils.find_alt_config(qe.reshape(-1, 1), limt6ontable, filterOriginalq=False).T
    print("qaltnew:", qaltnew)
