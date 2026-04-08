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
    groups_id = list(range(total_pairs))
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
    l = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    Q1 = find_alt_config2(q1, l)
    Q2 = find_alt_config2(q2, l)

    groups_pair, groups_num, total_pairs, groups_matrix = find_altconfig_redudancy(
        Q1, Q2
    )
    print("".center(50, "-"))
    print(f"==>> groups_matrix: \n{groups_matrix}")

    # need_cost = [True] * total_pairs
    # paths, costs = queue_altconfig_cost_estimation(
    #     groups_pair, groups_num, total_pairs, Q1, Q2, need_cost
    # )
    # print("Costs of unique pairs: \n", costs)


def _test_6d():
    q1 = np.random.uniform(-np.pi, np.pi, size=(6,))
    q2 = np.random.uniform(-np.pi, np.pi, size=(6,))
    l = np.array([[-2 * np.pi, 2 * np.pi]] * 6)
    Q1 = find_alt_config2(q1, l)
    Q2 = find_alt_config2(q2, l)

    groups_pair, groups_num, total_pairs, groups_matrix = find_altconfig_redudancy(
        Q1, Q2
    )
    # print("groups_pair groups: \n", groups_pair)
    # print("Number of each group: \n", groups_num)
    # print("Total number of pairs: \n", total_pairs)
    # print("Groups matrix: \n", groups_matrix)

    # need_cost = [True] * total_pairs
    # paths, costs = queue_altconfig_cost_estimation(
    #     groups_pair, groups_num, total_pairs, Q1, Q2, need_cost
    # )
    # print("Costs of unique pairs: \n", costs)


if __name__ == "__main__":
    _test_2d()
    # _test_6d()
