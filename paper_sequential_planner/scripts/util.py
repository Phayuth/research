import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import os
from functools import wraps
import time

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=2000)
rsrcpath = os.environ["RSRC_DIR"] + "/rnd_torus"


def simplify_tour(tour):
    """
    Turn = [(0, 3), (3, 6), (6, 4), (4, 9),..., (7, 1), (1, 0)]
    into simplified tour = [0, 3, 6, 4, 9, 5, 2, 8, 7, 1]
    """
    tour_simp = []
    for t in tour:
        tour_simp.append(t[0])
    return tour_simp


def rotate_tour_simplified_format(tour, startnodename):
    """
    rotate given tour to start at start_index
    tour must be in list: tour = [0, 3, 6, 4, 9, 5, 2, 8, 7, 1]
    """
    if isinstance(tour, np.ndarray):
        tour = tour.tolist()
    start_index = tour.index(startnodename)
    n = len(tour)
    return [tour[(i + start_index) % n] for i in range(n)]


def expand_tour(tour):
    """
    Turn simplified tour = [0, 3, 6, 4, 9, 5, 2, 8, 7, 1]
    tour = [(0, 3), (3, 6), (6, 4), (4, 9), ..., (8, 7), (7, 1), (1, 0)]
    """
    return [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)] + [
        (tour[-1], tour[0])
    ]


def rotate_tour_expanded_format(tour, startnodename):
    """
    rotate given tour to start at start_index
    tour must be in expanded format:
    tour = [(0, 3), (3, 6), (6, 4), (4, 9), ..., (8, 7), (7, 1), (1, 0)]
    """
    toursimp = simplify_tour(tour)
    rotatedtour = rotate_tour_simplified_format(toursimp, startnodename)
    return expand_tour(rotatedtour)


def H_to_xyzquat(H):
    xyz = H[:3, 3]
    r = R.from_matrix(H[:3, :3])
    quat = r.as_quat()  # [x,y,z,w]
    return xyz, quat


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


def nearest_neighbour_H(Hlist, hquery):
    xyz = []
    for i in range(len(Hlist)):
        xyz.append((Hlist[i][0, 3], Hlist[i][1, 3], Hlist[i][2, 3]))
    xyz = np.array(xyz)
    xyzquery = hquery[:3, 3]
    dists = np.linalg.norm(xyz - xyzquery, axis=1)
    nearest_idx = np.argmin(dists)
    return nearest_idx, Hlist[nearest_idx]


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
