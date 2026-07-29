import os
import numpy as np

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def weighted_nan_euclidean_distances(X, Y=None, w=None, dtype=np.float32):
    """
    Fully vectorized weighted NaN-aware Euclidean distance.

    Parameters
    ----------
    X : (n_samples_X, n_features)
    Y : (n_samples_Y, n_features), optional
    w : (n_features,), optional

    Returns
    -------
    D : (n_samples_X, n_samples_Y)
    """
    X = np.asarray(X, dtype=dtype)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=dtype)

    d = X.shape[1]

    if w is None:
        w = np.ones(d, dtype=dtype)
    else:
        w = np.asarray(w, dtype=dtype)

    total_weight = w.sum()

    # Valid entries
    valid = (~np.isnan(X))[:, None, :] & (~np.isnan(Y))[None, :, :]

    # Replace NaN by zero
    X0 = np.nan_to_num(X)
    Y0 = np.nan_to_num(Y)

    # Pairwise differences
    diff = X0[:, None, :] - Y0[None, :, :]

    # Weighted squared differences
    sqdist = np.sum(w * diff**2 * valid, axis=2)

    # Sum of observed weights
    observed = np.sum(w * valid, axis=2)

    # Normalize exactly like sklearn
    D = np.full_like(sqdist, np.nan, dtype=dtype)

    mask = observed > 0
    D[mask] = np.sqrt(sqdist[mask] * total_weight / observed[mask])

    return D


def weighted_nan_max_joint_diff_distances(X, Y=None, w=None, dtype=np.float32):
    """
    Pairwise weighted NaN-aware Chebyshev (maximum joint difference) distance.
                    |                 |
    d(x, y) = max_i | w_i*|x_i - y_i| |
                    |                 |
    Parameters
    ----------
    X : (n_samples_X, n_features)
    Y : (n_samples_Y, n_features), optional
    w : (n_features,), optional
        Positive scaling factors for each feature.

    Returns
    -------
    D : (n_samples_X, n_samples_Y)
    """
    X = np.asarray(X, dtype=dtype)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=dtype)

    d = X.shape[1]

    if w is None:
        w = np.ones(d, dtype=dtype)
    else:
        w = np.asarray(w, dtype=dtype)

    # Valid dimensions
    valid = (~np.isnan(X))[:, None, :] & (~np.isnan(Y))[None, :, :]

    # Pairwise absolute differences
    diff = np.abs(np.nan_to_num(X)[:, None, :] - np.nan_to_num(Y)[None, :, :])

    # Normalize by weights
    diff = w * diff

    # Ignore invalid dimensions
    diff = np.where(valid, diff, -np.inf)

    # Maximum over features
    D = diff.max(axis=2)

    # If no valid feature exists, return NaN
    D[np.all(~valid, axis=2)] = np.nan

    return D.astype(dtype)


def traj_mahattan_dist_cost(traj):
    """
    Compute the total Manhattan distance of a trajectory.
    traj: (N, d) array of points
    return: scalar total distance
    """
    return np.sum(np.sum(np.abs(np.diff(traj, axis=0)), axis=1))


def traj_euclidean_dist_cost(traj):
    """
    Compute the total Euclidean distance of a trajectory.
    traj: (N, d) array of points
    return: scalar total distance
    """
    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))


def traj_inf_dist_cost(traj):
    """
    Compute the total infinity norm distance of a trajectory.
    traj: (N, d) array of points
    return: scalar total distance
    """
    return np.sum(np.max(np.abs(np.diff(traj, axis=0)), axis=1))


def traj_perjoint_motion_cost(traj):
    """
    Compute the total per-joint motion cost of a trajectory.
    traj: (N, d) array of points
    return: vector of scalar total distance per joint
    """
    return np.sum(np.abs(np.diff(traj, axis=0)), axis=0)


def traj_time_cost(traj, qdot):
    """
    Simple time cost function
    Compute the total time cost of a trajectory given joint velocities.
    Equation is the same as inf norm distance, but normalized by joint velocities.
    traj: (N, d) array of points
    qdot: (d,) array of joint velocities
    return: scalar total time cost
    """
    return np.sum(np.max(np.abs(np.diff(traj, axis=0)) / qdot, axis=1))


def traj_complete_cost(traj):
    """
    Compute the complete cost view of a trajectory.
    traj: (N, d) array of points
    return: dict with various cost metrics
    """
    return {
        "manhattan": traj_mahattan_dist_cost(traj).item(),
        "euclidean": traj_euclidean_dist_cost(traj).item(),
        "inf": traj_inf_dist_cost(traj).item(),
        "perjoint": traj_perjoint_motion_cost(traj).tolist(),
    }
