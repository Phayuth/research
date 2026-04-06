import numpy as np
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances
import matplotlib.pyplot as plt


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


def find_alt_config_redudancy(Q1, Q2):
    """
    Find unique pairs of configurations between two sets of torus space
    >>> q1 = np.array([3.1, 0.1])
    >>> q2 = np.array([3.5, 1.0])
    >>> l = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    >>> Q1 = find_alt_config2(q1, l)
    >>> Q2 = find_alt_config2(q2, l)
    >>> unique_colors_ = plt.cm.get_cmap("tab10", len(value))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(Q1[:, 0], Q1[:, 1], "bo", label="Q1")
    >>> ax.plot(Q2[:, 0], Q2[:, 1], "ro", label="Q2")
    >>> for gi, group in enumerate(pairs_per_value):
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
    D = nan_euclidean_distances(Q1, Q2)
    D = np.round(D, decimals=5)  # round to handle floating-point precision issues
    value, idx, count = np.unique(D, return_index=True, return_counts=True)
    D_flat = D.ravel()
    inv = np.searchsorted(value, D_flat)
    groups = [np.where(inv == k)[0] for k in range(len(value))]
    groups_num = [len(g) for g in groups]
    total_pairs = len(groups_num)
    pairs_per_value = [
        np.column_stack(np.unravel_index(g, D.shape)) for g in groups
    ]
    return pairs_per_value, groups_num, total_pairs


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

    fig, ax = plt.subplots()
    ax.plot(qsog[0], qsog[1], "bo", label="qsog")
    ax.plot(path[:, 0], path[:, 1], "k.-", label="Original Path")
    ax.plot(pathnew[:, 0], pathnew[:, 1], "g.-", alpha=0.5, label="Path shifted")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.set_aspect("equal")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.legend()
    plt.show()


def transform_path_torus2(path, Q1redundantgroup):
    """
    Same version as transform_path_torus1 but transform to all the redundant pairs in the group.
    """
    qsog = path[0]
    diffall = Q1redundantgroup - qsog
    pathall = np.array([path + d for d in diffall])

    fig, ax = plt.subplots()
    ax.plot(qsog[0], qsog[1], "bo", label="qsog")
    ax.plot(path[:, 0], path[:, 1], "k.-", label="Original Path")
    for p in pathall:
        ax.plot(p[:, 0], p[:, 1], "g.-", alpha=0.5, label="Path shifted")
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.set_aspect("equal")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    q1 = np.array([3.1, 0.1])
    q2 = np.array([3.5, 1.0])
    l = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    Q1 = find_alt_config2(q1, l)
    Q2 = find_alt_config2(q2, l)
    print("Q1: \n", Q1)
    print("Q2: \n", Q2)

    redundant_pairs, groups_num, total_pairs = find_alt_config_redudancy(Q1, Q2)
    print("redundant_pairs groups: \n", redundant_pairs)
    print("Number of each group: \n", groups_num)
    print("Total number of pairs: \n", total_pairs)

    path = np.linspace(q1, q2, num=10)  # example path from q1 to q2
    path[1:-1] = path[1:-1] + np.random.rand(*path[1:-1].shape) * 0.1
    g1_id = redundant_pairs[0]
    Q1g1 = Q1[g1_id[:, 0]]
    transform_path_torus1(path, Q1g1[0])
    transform_path_torus2(path, Q1g1)
