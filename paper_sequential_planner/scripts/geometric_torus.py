import numpy as np
from itertools import product
from sklearn.metrics.pairwise import nan_euclidean_distances
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def map_val(x, inMin, inMax, outMin, outMax):
    """
    >>> m = 2
    >>> n = map_val(m, 0, 5, 0, 100)
    >>> n : 40.0

    >>> q = np.random.random((6, 1))
    >>> w = map_val(q, 0, 1, 0, 100)
    >>> w : [[ 4.21994177], [88.67016363],
    [64.28742299], [61.37314135],
    [18.75513219], [35.97633713]]
    """
    return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin


def wrap_to_pi(q):
    return (q + np.pi) % (2 * np.pi) - np.pi


def find_alt_config(
    q,
    configLimit,
    configConstrict=None,
    filterOriginalq=False,
):
    """
    Find the alternative configuration.

    configConstrict : Constrict specific joint from finding alternative.
    Ex: the last joint of robot doesn't make any different when moving so we ignore them.

    filterOriginalq : Filter out the original q value.
    keep only the alternative value in array.

    # 2 DOF
    >>> q2 = np.array([3.1, 0.0]).reshape(2, 1)
    >>> limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-np.pi, np.pi]])
    >>> gg = find_alt_config(q2, limt2)

    # 6 DOF
    >>> q6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1)
    >>> limt6 = np.array([[-2 * np.pi, 2 * np.pi],
                          [-2 * np.pi, 2 * np.pi],
                          [-np.pi, np.pi],
                          [-2 * np.pi, 2 * np.pi],
                          [-2 * np.pi, 2 * np.pi],
                          [-2 * np.pi, 2 * np.pi]])
    >>> const = [False, False, False, False, False, False]
    >>> u = find_alt_config(q6, limt6, const, filterOriginalq=False)
    """
    # possible config value
    qw = wrap_to_pi(q)  # transform to base quadrand first
    qcomb = np.array(
        list(product([-2.0 * np.pi, 0.0, 2.0 * np.pi], repeat=qw.shape[0]))
    ).T
    qShifted = qw + qcomb
    # eliminate with joint limit
    isInLimitMask = np.all(
        (qShifted >= configLimit[:, 0, np.newaxis])
        & (qShifted <= configLimit[:, 1, np.newaxis]),
        axis=0,
    )
    qInLimit = qShifted[:, isInLimitMask]

    # joint constrict
    if configConstrict is not None:
        assert isinstance(
            configConstrict, list
        ), "configConstrict must be in list format"
        assert (
            len(configConstrict) == qw.shape[0]
        ), "configConstrict length must be equal to state number"
        for i in range(len(configConstrict)):
            if configConstrict[i] is True:
                qInLimit[i] = qw[i]

    if filterOriginalq:
        # Use np.allclose with tolerance to handle floating-point precision errors
        exists = np.array(
            [
                np.allclose(qInLimit[:, i], q.flatten(), atol=1e-10, rtol=1e-10)
                for i in range(qInLimit.shape[1])
            ]
        )
        filterout = qInLimit[:, ~exists]
        return filterout

    return qInLimit


def find_alt_config2(q, configLimit, configConstrict=None, filterOriginalq=False):
    """
    The row vector version of find_alt_config. The output is in row vector format as well.
    >>> q2 = np.array([3.1, 1.0])
    >>> limt2 = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )
    gg = find_alt_config2(q2, limt2, filterOriginalq=False)
    """
    qw = wrap_to_pi(q)
    qcomb = np.array(
        list(product([-2.0 * np.pi, 0.0, 2.0 * np.pi], repeat=qw.shape[0]))
    )
    qShifted = qw + qcomb
    isInLimitMask = np.all(
        (qShifted >= configLimit[:, 0]) & (qShifted <= configLimit[:, 1]), axis=1
    )
    qInLimit = qShifted[isInLimitMask]

    if configConstrict is not None:
        assert isinstance(
            configConstrict, list
        ), "configConstrict must be in list format"
        assert (
            len(configConstrict) == qw.shape[0]
        ), "configConstrict length must be equal to state number"
        for i in range(len(configConstrict)):
            if configConstrict[i] is True:
                qInLimit[:, i] = qw[i]

    if filterOriginalq:
        exists = np.array(
            [
                np.allclose(qInLimit[i, :], q, atol=1e-10, rtol=1e-10)
                for i in range(qInLimit.shape[0])
            ]
        )
        filterout = qInLimit[~exists]
        return filterout

    return qInLimit


def sort_config(qs, qAlts):
    dist = np.linalg.norm(qAlts - qs, axis=0)
    return np.argsort(dist)


def minimum_dist_torus(qa, qb):
    """
    >>> qa = np.array([3.1, 0.0]).reshape(2, 1)
    >>> qb = np.array([-3.1, 0.0]).reshape(2, 1)
    >>> aa = minimum_dist_torus(qa, qb)
    >>> aa : 0.08318530717958605
    """
    L = np.full_like(qa, 2 * np.pi)
    delta = np.abs(qa - qb)
    deltaw = L - delta
    deltat = np.min(np.hstack((delta, deltaw)), axis=1)
    return np.linalg.norm(deltat)


def find_altconfig_redudancy(Q1, Q2, Dist=None):
    """
    When q1 and q2 delta distance of each joint have the same value
    I get the wrong result.
    Right result for the wrong reason. must make new function
    Find unique pairs of configurations between two sets of torus space
    unique pairs = 3^dof
    2d have 9 unique pairs
    3d have 27 unique pairs
    4d have 81 unique pairs
    5d have 243 unique pairs
    6d have 729 unique pairs
    >>> q1 = np.array([3.1, 0.1])
    >>> q2 = np.array([3.5, 1.0])
    >>> l = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    >>> Q1 = find_alt_config2(q1, l)
    >>> Q2 = find_alt_config2(q2, l)
    >>> unique_colors_ = plt.cm.get_cmap("tab10", len(value))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(Q1[:, 0], Q1[:, 1], "bo", label="Q1")
    >>> ax.plot(Q2[:, 0], Q2[:, 1], "ro", label="Q2")
    >>> for gi, group in enumerate(groups_pair):
    >>>     for i, j in group:
    >>>         ax.plot(
    >>>             [Q1[i, 0], Q2[j, 0]],
    >>>             [Q1[i, 1], Q2[j, 1]],
    >>>             "--",
    >>>             color=unique_colors_(gi),
    >>>         )
    >>> ax.set_xlim(-2 * np.pi, 2 * np.pi)
    >>> ax.set_ylim(-2 * np.pi, 2 * np.pi)
    >>> ax.set_aspect("equal")
    >>> ax.legend()
    >>> plt.show()
    """
    if Dist is None:
        Dist = nan_euclidean_distances(Q1, Q2)
    Dist = np.round(Dist, decimals=5)  # round to handle floating-point issues
    unq_val, idx, groups_matrix, groups_num = np.unique(
        Dist, return_index=True, return_inverse=True, return_counts=True
    )
    total_pairs = len(groups_num)
    group_pairs = [np.where(groups_matrix == k) for k in range(total_pairs)]
    group_pairs = [np.column_stack(gp) for gp in group_pairs]
    return group_pairs, groups_num, total_pairs, groups_matrix


def find_altconfig_redudancy2(Q1, Q2, q1, q2):
    """
    Same version as find_altconfig_redudancy but fixing the wrong result
    """
    diff1 = Q1 - q1
    diff2 = Q2 - q2
    unique_diffs = set()
    group_id = np.zeros((Q1.shape[0], Q2.shape[0]), dtype=int)
    for d1 in diff1:
        for d2 in diff2:
            unique_diffs.add(tuple((d2 - d1).round(5).tolist()))
    for i in range(Q1.shape[0]):
        for j in range(Q2.shape[0]):
            d1 = diff1[i]
            d2 = diff2[j]
            delta = tuple((d2 - d1).round(5).tolist())
            group_id[i, j] = list(unique_diffs).index(delta)

    unq_val, idx, groups_matrix, groups_num = np.unique(
        group_id, return_index=True, return_inverse=True, return_counts=True
    )
    total_pairs = len(groups_num)
    group_pairs = [np.where(groups_matrix == k) for k in range(total_pairs)]
    group_pairs = [np.column_stack(gp) for gp in group_pairs]
    return group_pairs, groups_num, total_pairs, groups_matrix


def transform_path_torus1(path, qs_new):
    """
    Path must start with q1, and qs_new must be the alternative configuration of q1.
    Transform a path found on a torus pair to it redudancy group.
    Technically I don't need to transform the path to all the redundant pairs
    It waste time, just propagrate the cost
    When the path is finalized and ask for the actual path, I transform later.
    """
    qsog = path[0]
    diff = qs_new - qsog
    pathnew = path + diff
    return pathnew  # return a path shaped (N, dof) same as old path


def transform_path_torus2(path, Q1redundantgroup):
    """
    Same version as transform_path_torus1 but transform to all the redundant pairs in the group.
    path old shape is (N, dof), Q1redundantgroup shape is (M, dof), output is (M, N, dof)

    >>> q1 = np.array([3.1, 0.1])
    >>> q2 = np.array([3.5, 1.0])
    >>> l = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    >>> Q1 = find_alt_config2(q1, l)
    >>> Q2 = find_alt_config2(q2, l)
    >>> path = _generate_fake_path(q1, q2, num_points=10)
    >>> g1_id = groups_pair[0]
    >>> Q1g1 = Q1[g1_id[:, 0]]
    >>> pathall = transform_path_torus2(path, Q1g1)
    >>> print(f"==>> pathall.shape: \n{pathall.shape}")

    >>> fig, ax = plt.subplots()
    >>> ax.plot(path[:, 0], path[:, 1], "k.-", label="Original Path")
    >>> for p in pathall:
    >>>     ax.plot(p[:, 0], p[:, 1], "g.-", alpha=0.5, label="Path shifted")
    >>> ax.set_xlim(-2 * np.pi, 2 * np.pi)
    >>> ax.set_ylim(-2 * np.pi, 2 * np.pi)
    >>> ax.set_aspect("equal")
    >>> ax.set_xlabel("q1")
    >>> ax.set_ylabel("q2")
    >>> ax.legend()
    >>> plt.show()

    """
    qsog = path[0]
    diffall = Q1redundantgroup - qsog
    pathall = path[np.newaxis, :, :] + diffall[:, np.newaxis, :]
    return pathall  # return a path shaped (M, N, dof) same as old path


def queue_altconfig_cost_estimation(
    groups_pair, groups_num, total_pairs, Q1, Q2, need_cost
):
    """
    Queue for cost estimation of all unique pairs.
    We get the qs and qg from the first pair of each group
    need_cost is a list of bool, same length as total_pairs, if we need cost
    """
    first_pair_id = [re[0] for re in groups_pair]
    qsqg_pairs = [(Q1[i], Q2[j]) for i, j in first_pair_id]

    paths = [None] * total_pairs  # initialize path list
    costs = [None] * total_pairs  # initialize cost list
    for i in range(total_pairs):
        if need_cost[i]:
            qs, qg = qsqg_pairs[i]
            p, c = _generate_fake_path(qs, qg)  # replace later
            costs[i] = c
            paths[i] = p
        else:
            costs[i] = None
            paths[i] = None
    return paths, costs


def _generate_fake_path(q1, q2, num_points=10):
    """Generate a fake path from q1 to q2 with some noise for testing."""
    path = np.linspace(q1, q2, num=num_points)  # example path from q1 to q2
    path[1:-1] = path[1:-1] + np.random.rand(*path[1:-1].shape) * 0.1

    diff = np.diff(path, axis=0)
    cost = np.sum(np.linalg.norm(diff, axis=1))
    return path, cost


def _test_2d():
    q1 = np.array([3.1, 0.1])
    q2 = np.array([3.5, 1.0])
    q11 = np.array([3.1, 0.1])  # all delta is 0.4 the same so I get wrong result
    q22 = np.array([3.5, 0.5])
    l = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    Q1 = find_alt_config2(q1, l)
    Q2 = find_alt_config2(q2, l)
    Q11 = find_alt_config2(q11, l)
    Q22 = find_alt_config2(q22, l)

    print("".center(50, "-"))
    g_pair_c, g_num_c, total_pairs_c, g_matrix_c = find_altconfig_redudancy2(
        Q1, Q2, q1, q2
    )
    print(f"==>> g_pair_c: \n{g_pair_c}")
    print(f"==>> g_num_c: \n{g_num_c}")
    print(f"==>> total_pairs_c: \n{total_pairs_c}")
    print(f"==>> g_matrix_c: \n{g_matrix_c}")
    print("".center(50, "-"))
    g_pair, g_num, total_pairs, g_matrix = find_altconfig_redudancy(Q1, Q2)
    print(f"==>> g_pair: \n{g_pair}")
    print(f"==>> g_num: \n{g_num}")
    print(f"==>> total_pairs: \n{total_pairs}")
    print(f"==>> g_matrix: \n{g_matrix}")
    print("".center(50, "-"))

    print("".center(50, "-"))
    g_pair_c1, g_num_c1, total_pairs_c1, g_matrix_c1 = find_altconfig_redudancy2(
        Q11, Q22, q11, q22
    )
    print(f"==>> g_pair_c1: \n{g_pair_c1}")
    print(f"==>> g_num_c1: \n{g_num_c1}")
    print(f"==>> total_pairs_c1: \n{total_pairs_c1}")
    print(f"==>> g_matrix_c1: \n{g_matrix_c1}")
    print("".center(50, "-"))
    g_pair1, g_num1, total_pairs1, g_matrix1 = find_altconfig_redudancy(Q11, Q22)
    print(f"==>> g_pair: \n{g_pair1}")
    print(f"==>> g_num: \n{g_num1}")
    print(f"==>> total_pairs: \n{total_pairs1}")
    print(f"==>> g_matrix: \n{g_matrix1}")
    print("".center(50, "-"))

    unique_colors_1 = plt.cm.get_cmap("tab10", total_pairs)
    unique_colors_2 = plt.cm.get_cmap("tab10", total_pairs1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(Q1[:, 0], Q1[:, 1], "bo", label="Q1")
    ax1.plot(Q2[:, 0], Q2[:, 1], "ro", label="Q2")
    for gi, group in enumerate(g_pair):
        for i, j in group:
            ax1.plot(
                [Q1[i, 0], Q2[j, 0]],
                [Q1[i, 1], Q2[j, 1]],
                "--",
                color=unique_colors_1(gi),
                label=(
                    f"Group {gi}"
                    if i == group[0][0] and j == group[0][1]
                    else None
                ),
            )
    ax1.set_xlim(-2 * np.pi, 2 * np.pi)
    ax1.set_ylim(-2 * np.pi, 2 * np.pi)
    ax1.set_aspect("equal")
    ax1.set_title("Correct Redundancy when delta is different")

    ax2.plot(Q11[:, 0], Q11[:, 1], "bo", label="Q11")
    ax2.plot(Q22[:, 0], Q22[:, 1], "ro", label="Q22")
    for gi, group in enumerate(g_pair1):
        for i, j in group:
            ax2.plot(
                [Q11[i, 0], Q22[j, 0]],
                [Q11[i, 1], Q22[j, 1]],
                "--",
                color=unique_colors_2(gi),
                label=(
                    f"Group {gi}"
                    if i == group[0][0] and j == group[0][1]
                    else None
                ),
            )

    # correct result
    unique_colors_11 = plt.cm.get_cmap("tab10", total_pairs_c)
    unique_colors_22 = plt.cm.get_cmap("tab10", total_pairs_c1)
    ax3.plot(Q1[:, 0], Q1[:, 1], "bo", label="Q1")
    ax3.plot(Q2[:, 0], Q2[:, 1], "ro", label="Q2")
    for gi, group in enumerate(g_pair_c):
        for i, j in group:
            ax3.plot(
                [Q1[i, 0], Q2[j, 0]],
                [Q1[i, 1], Q2[j, 1]],
                "--",
                color=unique_colors_11(gi),
                label=(
                    f"Group {gi}"
                    if i == group[0][0] and j == group[0][1]
                    else None
                ),
            )

    ax4.plot(Q11[:, 0], Q11[:, 1], "bo", label="Q11")
    ax4.plot(Q22[:, 0], Q22[:, 1], "ro", label="Q22")
    for gi, group in enumerate(g_pair_c1):
        for i, j in group:
            ax4.plot(
                [Q11[i, 0], Q22[j, 0]],
                [Q11[i, 1], Q22[j, 1]],
                "--",
                color=unique_colors_22(gi),
                label=(
                    f"Group {gi}"
                    if i == group[0][0] and j == group[0][1]
                    else None
                ),
            )
    ax2.set_xlim(-2 * np.pi, 2 * np.pi)
    ax2.set_ylim(-2 * np.pi, 2 * np.pi)
    ax2.set_aspect("equal")
    ax2.set_title("Wrong Lost of redundancy when delta is the same")
    ax2.legend()

    plt.show()


def _test_6d():
    q1 = np.array([0.5, -0.2, 0.3, -0.4, 0.5, -0.6])
    q2 = np.array([0.9, -0.9, 0.4, 0.5, 0.9, 1.1])
    # q1 = np.random.uniform(-np.pi, np.pi, size=6)
    # q2 = np.random.uniform(-np.pi, np.pi, size=6)
    lall2pi = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )
    Q1all2pi = find_alt_config2(q1, lall2pi)
    Q2all2pi = find_alt_config2(q2, lall2pi)
    print(f"==>> Q1all2pi.shape: \n{Q1all2pi.shape}")
    print(f"==>> Q2all2pi.shape: \n{Q2all2pi.shape}")

    find_altconfig_redudancy2(Q1all2pi, Q2all2pi, q1, q2)

    groups_pair, groups_num, total_pairs, groups_matrix = find_altconfig_redudancy(
        Q1all2pi, Q2all2pi
    )
    # print("groups_pair groups: \n", groups_pair)
    # print("Number of each group: \n", groups_num)
    print("Total number of pairs: \n", total_pairs)
    # print("Groups matrix: \n", groups_matrix)

    print("".center(50, "-"))
    lphysical = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-np.pi, np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )
    Q1physical = find_alt_config2(q1, lphysical)
    Q2physical = find_alt_config2(q2, lphysical)
    print(f"==>> Q1physical.shape: \n{Q1physical.shape}")
    print(f"==>> Q2physical.shape: \n{Q2physical.shape}")

    find_altconfig_redudancy2(Q1physical, Q2physical, q1, q2)

    groups_pair, groups_num, total_pairs, groups_matrix = find_altconfig_redudancy(
        Q1physical, Q2physical
    )
    # print("groups_pair groups: \n", groups_pair)
    print("Number of each group: \n", groups_num)
    print("Total number of pairs: \n", total_pairs)
    # print("Groups matrix: \n", groups_matrix)
    print("".center(50, "-"))

    lontable = np.array(
        [
            [-2 * np.pi, 2 * np.pi],
            [-np.pi, 0],
            [-np.pi, np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]
    )

    Q1ontable = find_alt_config2(q1, lontable)
    print(f"==>> Q1ontable.shape: \n{Q1ontable.shape}")
    Q2ontable = find_alt_config2(q2, lontable)
    print(f"==>> Q2ontable.shape: \n{Q2ontable.shape}")

    find_altconfig_redudancy2(Q1ontable, Q2ontable, q1, q2)

    groups_pair, groups_num, total_pairs, groups_matrix = find_altconfig_redudancy(
        Q1ontable, Q2ontable
    )
    # print("groups_pair groups: \n", groups_pair)
    print("Number of each group: \n", groups_num)
    print("Total number of pairs: \n", total_pairs)
    # print("Groups matrix: \n", groups_matrix)
    print("".center(50, "-"))


if __name__ == "__main__":
    _test_2d()
    # _test_6d()
