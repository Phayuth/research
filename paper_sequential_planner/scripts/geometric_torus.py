import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances
from itertools import product


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
    D = nan_euclidean_distances(Q1, Q2)
    print(f"==>> D: \n{D}")
    u = np.unique(D)
    print(f"==>> u: \n{u}")


if __name__ == "__main__":
    q1 = np.array([3.1, 0.1])
    q2 = np.array([3.1, 1.0])
    l = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
    Q1 = find_alt_config2(q1, l)
    Q2 = find_alt_config2(q2, l)
    print("Q1: \n", Q1)
    print("Q2: \n", Q2)
    redundancy = find_alt_config_redudancy(Q1, Q2)
