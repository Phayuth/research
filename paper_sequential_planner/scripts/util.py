import numpy as np
from spatial_geometry.utils import Utils
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis
import pickle
import os
from functools import wraps
import time

try:
    from eaik.IK_DH import DhRobot
    from spatialmath import SE3
    from roboticstoolbox import DHRobot, RevoluteDH
except:
    print("missing packages; usage limited")

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=2000)
rsrcpath = os.environ["RSRC_DIR"] + "/rnd_torus"


def ur5e_dh():
    d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
    alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
    a = np.array([0, -0.425, -0.3922, 0, 0, 0])
    bot = DhRobot(alpha, a, d)
    return bot


def ur5e_rtb_dh():
    L1 = RevoluteDH(d=0.1625, a=0, alpha=np.pi / 2, qlim=[-2 * np.pi, 2 * np.pi])
    L2 = RevoluteDH(d=0, a=-0.425, alpha=0, qlim=[-2 * np.pi, 2 * np.pi])
    L3 = RevoluteDH(d=0, a=-0.3922, alpha=0, qlim=[-np.pi, np.pi])
    L4 = RevoluteDH(d=0.1333, a=0, alpha=np.pi / 2, qlim=[-2 * np.pi, 2 * np.pi])
    L5 = RevoluteDH(d=0.0997, a=0, alpha=-np.pi / 2, qlim=[-2 * np.pi, 2 * np.pi])
    L6 = RevoluteDH(d=0.0996, a=0, alpha=0, qlim=[-2 * np.pi, 2 * np.pi])
    ur5e = DHRobot([L1, L2, L3, L4, L5, L6], name="UR5e")
    return ur5e


def _verify_bot_():

    bot = ur5e_dh()
    botrtb = ur5e_rtb_dh()

    Q = np.random.uniform(-np.pi, np.pi, (10, 6))
    for i in range(Q.shape[0]):
        q = Q[i]
        H = botrtb.fkine(q)
        H1 = solve_fk(bot, q)
        # print(H.A)
        # print(H1)
        print("is equal", np.allclose(H.A, H1))
        print("-----")


def solve_fk(bot, angles):
    return bot.fwdKin(angles)


def solve_ik(bot, h):
    sols = bot.IK(h)
    return sols.num_solutions(), sols.Q


def solve_ik_bulk(bot, H):
    num_sols = []
    ik_sols = []
    for h in H:
        num_sol, ik_sol = solve_ik(bot, h)
        num_sols.append(num_sol)
        ik_sols.append(ik_sol)
    return num_sols, ik_sols


def solve_ik_altconfig(bot, h):
    numsol, Qik = solve_ik(bot, h)
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
    ikaltconfig = []
    for i in range(numsol):
        alt = Utils.find_alt_config(Qik[i].reshape(6, 1), limt6)
        ikaltconfig.append(alt)
    ikaltconfig = np.hstack(ikaltconfig).T
    return ikaltconfig.shape[0], ikaltconfig


def solve_ik_altconfig_bulk(bot, H):
    num_sols = []
    ik_sols = []
    for h in H:
        num_sol, ik_solaltconfig = solve_ik_altconfig(bot, h)
        num_sols.append(num_sol)
        ik_sols.append(ik_solaltconfig)
    return num_sols, ik_sols


def find_altconfig(Q):
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
    altconfig = []
    for i in range(Q.shape[0]):
        alt = Utils.find_alt_config(Q[i].reshape(6, 1), limt6)
        altconfig.append(alt)
    altconfig = np.hstack(altconfig).T
    return altconfig.shape[0], altconfig


def find_altconfig_bulk(Qlist):
    num_sols = []
    alt_sols = []
    for Q in Qlist:
        num_sol, alt_sol = find_altconfig(Q)
        num_sols.append(num_sol)
        alt_sols.append(alt_sol)
    return num_sols, alt_sols


def convert_urdf_to_dh_frame(H):
    "from our design task in urdf frame to dh frame"
    Hdh_to_urdf = SE3.Rz(np.pi).A
    return np.linalg.inv(Hdh_to_urdf) @ H


def generate_random_dh_tasks(bot, num_tasks=10):
    angle = np.random.uniform(-np.pi, np.pi, size=(num_tasks, 6))
    T = []
    for i in range(num_tasks):
        t = solve_fk(bot, angle[i])
        T.append(t)
    return T


def generate_random_task_transformation():
    translation = np.random.uniform(-1, 1, size=(3,))
    rotation = np.random.uniform(-np.pi, np.pi, size=(4,))
    transformation = np.eye(4)
    RR = R.from_quat(rotation)
    transformation[:3, :3] = RR.as_matrix()
    transformation[:3, 3] = translation
    return transformation


def generate_linear_tasks_transformation(
    s=[1, 1, 1],
    e=[1, -1, 1],
    quat=[0.0, 0.707106, 0.0, 0.707106],
    num_tasks=10,
):
    t = np.linspace(s, e, num_tasks)
    Hlist = [np.eye(4) for _ in range(num_tasks)]
    for i in range(num_tasks):
        Hlist[i][:3, 3] = t[i]
        Hlist[i][:3, :3] = R.from_quat(quat).as_matrix()
    return Hlist


def generate_linear_grid_tasks_transformation():
    size = 4
    H1 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.6], [0.5, -0.5, 0.6], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H2 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.4], [0.5, -0.5, 0.4], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H3 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.2], [0.5, -0.5, 0.2], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H4 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.707106, 0.0, 0.707106], size
    )
    Hlist = H1 + H2 + H3 + H4
    for i in range(len(Hlist)):
        Hlist[i] = convert_urdf_to_dh_frame(Hlist[i])
    return Hlist


def generate_linear_dual_side_tasks_transformation():
    size = 4
    H1 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.6], [0.5, -0.5, 0.6], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H2 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.4], [0.5, -0.5, 0.4], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H3 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.2], [0.5, -0.5, 0.2], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H4 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.707106, 0.0, 0.707106], size
    )
    Hlist = H1 + H2 + H3 + H4
    H1list = []
    for h in Hlist:
        H1list.append(convert_urdf_to_dh_frame(h))
    return Hlist + H1list


def generate_spiral_task_transformation():
    turns = 5  # number of full rotations
    points_per_turn = 5  # resolution along the curve
    height = 0.7  # total height of the spiral
    radius = 0.5  # constant radius (for a helix). Change below for conical

    # Parametric variable
    t = np.linspace(0, 2 * np.pi * turns, points_per_turn * turns)

    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = (height / (2 * np.pi * turns)) * t

    H = []
    for i in range(len(x)):
        Hi = np.eye(4)
        Hi[:3, 3] = np.array([x[i], y[i], z[i]])
        Hi[:3, :3] = R.from_quat([0.0, 0.707106, 0.0, 0.707106]).as_matrix()
        H.append(Hi)

    basisvector = np.array([1, 0])
    Hx = []
    for i in range(len(x)):
        xyvector = np.array([x[i], y[i]])
        alpha = np.arccos(np.dot(basisvector, xyvector) / np.linalg.norm(xyvector))
        Hx.append(SE3.Rx(alpha).A)

    for i in range(len(H)):
        H[i] = H[i] @ Hx[i]
    return H


def generate_ndarray():
    ndarray = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.5, 0.2],
            [0.7, 0.8],
        ]
    )
    return ndarray


def generate_ndarray_random():
    ndarray = np.random.uniform(-1, 1, size=(30, 2))
    return ndarray


def pickle_dump(obj, filename):
    file = os.path.join(rsrcpath, filename)
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    file = os.path.join(rsrcpath, filename)
    with open(file, "rb") as f:
        return pickle.load(f)


def np_save(npobj, filename):
    file = os.path.join(rsrcpath, filename)
    np.save(file, npobj)


def np_load(filename):
    file = os.path.join(rsrcpath, filename)
    npobj = np.load(file)
    return npobj


def np_load_csv(filename):
    file = os.path.join(rsrcpath, filename)
    npobj = np.loadtxt(file, delimiter=",")
    return npobj


def simplify_tour(tour):
    """
    Turn = [(0, 3), (3, 6), (6, 4), (4, 9),..., (7, 1), (1, 0)]
    into simplified tour = [0, 3, 6, 4, 9, 5, 2, 8, 7, 1]
    """
    tour_simp = []
    for t in tour:
        tour_simp.append(t[0])
    return tour_simp


def expand_tour(tour):
    """
    Turn simplified tour = [0, 3, 6, 4, 9, 5, 2, 8, 7, 1]
    tour = [(0, 3), (3, 6), (6, 4), (4, 9), ..., (8, 7), (7, 1), (1, 0)]
    """
    return [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)] + [
        (tour[-1], tour[0])
    ]


def rotate_tour_coords_format(tour, startnodename):
    """
    rotate given tour to start at start_index
    tour must be in coords:
    tour = [(0, 3), (3, 6), (6, 4), (4, 9), ..., (8, 7), (7, 1), (1, 0)]
    """
    toursimp = simplify_tour(tour)
    rotatedtour = rotate_tour_simplifiy_format(toursimp, startnodename)
    return expand_tour(rotatedtour)


def rotate_tour_simplifiy_format(tour, startnodename):
    """
    rotate given tour to start at start_index
    tour must be in list: tour = [0, 3, 6, 4, 9, 5, 2, 8, 7, 1]
    """
    if isinstance(tour, np.ndarray):
        tour = tour.tolist()
    start_index = tour.index(startnodename)
    n = len(tour)
    return [tour[(i + start_index) % n] for i in range(n)]


def tf_to_xyzquat(H):
    xyz = H[:3, 3]
    r = R.from_matrix(H[:3, :3])
    quat = r.as_quat()  # [x,y,z,w]
    return xyz, quat


def make_coords_from_ndarray(ndarray):
    coords = {i: (x, y) for i, (x, y) in enumerate(ndarray)}
    return coords


def make_ndarray_from_coords(coords):
    coords = []
    for i in range(len(coords)):
        coords.append((coords[i][0], coords[i][1]))
    return np.array(coords)


def make_coords_from_tasks_list(taskH):
    """
    Create a dictionary of coordinates from the task list.
    T = [H1, H2, ..., Hn] where Hi is a 4x4 transformation matrix.
    Example: {0: (x0, y0, z0),
              1: (x1, y1, z1),
              n: (xn, yn, zn)}
    """
    coords = {
        i: (taskH[i][0, 3], taskH[i][1, 3], taskH[i][2, 3])
        for i in range(len(taskH))
    }
    return coords


def make_ndarray_from_task_list(taskH):
    """
    T = [H1, H2, ..., Hn]
    Example: np.array([[x,y,z],
                       [x,y,z],
                       [x,y,z]])
    """
    xyz = []
    for i in range(len(taskH)):
        xyz.append((taskH[i][0, 3], taskH[i][1, 3], taskH[i][2, 3]))
    return np.array(xyz)


def generate_adjmatrix_seq_matter():
    task_candidates_num = np.array([0, 1, 4, 4, 4])
    node_num = task_candidates_num.sum()
    cs = np.cumsum(task_candidates_num)
    numtask = task_candidates_num.shape[0] - 2

    adj_matrix = np.zeros((node_num, node_num))

    for k in range(numtask):
        for i in range(cs[k], cs[k + 1]):
            for j in range(cs[k + 1], cs[k + 2]):
                adj_matrix[i, j] = 1

    print(f"adj_matrix:")
    print(adj_matrix)
    return adj_matrix


def generate_adjmatrix_seq_doesnt_matter():
    task_candidates_num = np.array([0, 1, 4, 4, 4])
    node_num = task_candidates_num.sum()
    cs = np.cumsum(task_candidates_num)
    numtask = task_candidates_num.shape[0] - 2

    adj_matrix = np.ones((node_num, node_num))
    for k in range(numtask + 1):
        for i in range(cs[k], cs[k + 1]):
            for j in range(cs[k], cs[k + 1]):
                adj_matrix[i, j] = 0

    print(f"adj_matrix:")
    print(adj_matrix)
    return adj_matrix


def generate_adjmatrix():
    graph = np.array(
        [
            [0, 35, 30, 20, 0, 0, 0],
            [35, 0, 8, 0, 12, 0, 0],
            [30, 8, 0, 9, 10, 20, 0],
            [20, 0, 9, 0, 0, 0, 15],
            [0, 12, 10, 0, 0, 5, 20],
            [0, 0, 20, 0, 5, 0, 5],
            [0, 0, 0, 15, 20, 5, 0],
        ]
    )
    return graph


def path_configuration_length(path):
    """
    compute the length of the path configuration
    ex: path = np.array(
        [[0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],])
    length = 2
    """
    diff = np.diff(path, axis=0)
    lengths = np.linalg.norm(diff, axis=1)
    return lengths.sum()


def nearest_neighbour_transformation(Hlist, hquery):
    xyz = []
    for i in range(len(Hlist)):
        xyz.append((Hlist[i][0, 3], Hlist[i][1, 3], Hlist[i][2, 3]))
    xyz = np.array(xyz)
    xyzquery = hquery[:3, 3]
    dists = np.linalg.norm(xyz - xyzquery, axis=1)
    nearest_idx = np.argmin(dists)
    return nearest_idx, Hlist[nearest_idx]


# ========================


def times_zero_to_one(numseg):
    time = np.linspace(0, 1, num=numseg)
    return time


def plot_joint_times(joint_path, times, joint_path_aux=None):
    numseg, numjoints = joint_path.shape
    fig, axs = plt.subplots(numjoints, 1, sharex=True)
    for i in range(numjoints):
        axs[i].plot(
            times,
            joint_path[:, i],
            color="blue",
            marker="o",
            linestyle="dashed",
            linewidth=2,
            markersize=6,
            label=f"Position",
        )
    if joint_path_aux is not None:
        for i in range(numjoints):
            axs[i].plot(
                times,
                joint_path_aux[:, i],
                color="orange",
                marker="o",
                linestyle="dashed",
                linewidth=2,
                markersize=6,
                label=f"Position (Aux)",
            )

    # visual setup
    for i in range(numjoints):
        axs[i].set_ylabel(f"Joint {i+1}")
        axs[i].set_xlim(times[0], times[-1])
        axs[i].set_ylim(-np.pi, np.pi)
        axs[i].legend(loc="upper right")
        axs[i].grid(True)
    axs[-1].set_xlabel("Time")
    plt.show()


# ======================== MISC ========================= #
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # Call the original function
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        # Return both the original result and the time taken
        return result, time_taken

    return wrapper


def option_runner(funclist, default_id="1"):
    for i, f in enumerate(funclist, start=1):
        print(f"{i}: {f.__name__}")

    if default_id is not None:
        arg = default_id
    else:
        arg = input("Enter argument number (` to exit): ")

    if arg == "`":
        print("Exiting...")
    elif arg.isdigit() and 1 <= int(arg) <= len(funclist):
        funclist[int(arg) - 1]()


# ========================= Visualization ======================== #
def plot_tf(T, names):
    ax = make_3d_axis(ax_s=1, unit="m")
    plot_transform(ax=ax, s=0.5, name="base_frame")  # basis
    if isinstance(T, list):
        for i, t in enumerate(T):
            if names is not None:
                name = names[i]
            else:
                name = f"{i}"
            plot_transform(ax=ax, A2B=t, s=0.1, name=name)
    else:
        plot_transform(ax=ax, A2B=T, s=0.1, name="frame")
    return ax


def plot_tf_tour(T, names, tour):
    ax = plot_tf(T, names)
    if tour is not None:
        for i, j in tour:
            ax.plot(
                [T[i][0, 3], T[j][0, 3]],
                [T[i][1, 3], T[j][1, 3]],
                [T[i][2, 3], T[j][2, 3]],
                color="red",
                linewidth=2,
            )
    ax.set_title(f"Tour: {tour}")
    return ax


def plot_2d_tour(coords, tour):
    """
    coords: (n,2) array of xy points
    tour: (n,) array of city indices
    """
    ordered_coords = coords[tour]

    # close the loop
    closed_coords = np.vstack([ordered_coords, ordered_coords[0]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(closed_coords[:, 0], closed_coords[:, 1], "-o", markersize=8)
    for i, (x, y) in enumerate(coords):
        ax.text(x + 0.02, y + 0.02, str(i), fontsize=9)
    return ax


def plot_2d_tour_coord(coords, tour):
    """for tsp"""
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, (x, y) in coords.items():
        ax.plot(x, y, "bo")
        ax.text(x + 0.2, y + 0.2, str(i), fontsize=12)

    for i, j in tour:
        xi, yi = coords[i]
        xj, yj = coords[j]
        ax.plot([xi, xj], [yi, yj], "r-")

    ax.set_title(f"Tour: {tour}")
    ax.axis("equal")
