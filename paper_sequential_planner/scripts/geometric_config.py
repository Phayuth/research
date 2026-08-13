import os
import numpy as np

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


def weighted_nan_euclidean_squared_distances(X, Y=None, w=None, dtype=np.float32):
    """
    Fully vectorized weighted NaN-aware squared Euclidean distance.

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
    D[mask] = sqdist[mask] * total_weight / observed[mask]

    return D


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
    Equation is the same as Manhattan distance, but computed per joint.

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


def traj_complete_cost(traj, *args, **kwargs):
    """
    Compute the complete cost view of a trajectory.

    traj: (N, d) array of points
    return: dict with various cost metrics
    """
    return {
        "manhattan": traj_mahattan_dist_cost(traj).item(),
        "euclidean": traj_euclidean_dist_cost(traj).item(),
        "inf": traj_inf_dist_cost(traj).item(),
        "time": traj_time_cost(traj, *args, **kwargs).item(),
        "perjoint": traj_perjoint_motion_cost(traj).tolist(),
    }


def batch_config_pairwise_lininterp(Q1, Q2, num_points):
    """
    Pairwise linear interpolation between corresponding rows of two sets of configurations.
    Q1: (n, dof)
    Q2: (n, dof)
    num_points: number of interpolation points (including endpoints)
    return: (n, n, num_points, dof)
    """
    # Reshape for broadcasting: (n, 1, 1, dof) and (1, n, 1, dof)
    Q1_exp = Q1[:, np.newaxis, np.newaxis, :]
    Q2_exp = Q2[np.newaxis, :, np.newaxis, :]
    # Interpolation parameter: (1, 1, num_points, 1)
    t = np.linspace(0, 1, num_points)[np.newaxis, np.newaxis, :, np.newaxis]
    # Linear interpolation with full broadcasting
    result = Q1_exp + (Q2_exp - Q1_exp) * t  # (n, n, num_points, dof)
    # Propagate NaN: if Q1 or Q2 has NaN, result is NaN
    result = np.where(np.isnan(Q1_exp) | np.isnan(Q2_exp), np.nan, result)
    return result


def batch_config_pairwise_center(Q1, Q2):
    """
    Compute the center configuration between two sets of configurations.
    Q1: (n, dof)
    Q2: (n, dof)
    return: (n, n, dof)
    """
    Qcenter = (Q1[:, np.newaxis, :] + Q2[np.newaxis, :, :]) / 2.0  # (n, n, dof)
    return Qcenter
    # Qcenter = np.where(np.isnan(Q1) | np.isnan(Q2), np.nan, (Q1 + Q2) / 2.0)
    # return Qcenter


def traj_tour_from_lininterp(Q, num_points, dt=0.1):
    """
    Creating tour trajectory given the sequence of waypoints configurations Q.
    Each segment between two waypoints is fixed to have num_points points.
    This make some segment to look longer than others.

    Q: (n, dof)
    return: (n * num_points, dof)
    """
    traj = np.empty((Q.shape[0] - 1, num_points, Q.shape[1]))
    time_from_start = np.arange(traj.shape[0]) * dt
    for i in range(Q.shape[0] - 1):
        traj[i] = np.linspace(Q[i], Q[i + 1], num_points)
    traj = traj.reshape(-1, Q.shape[1])
    time_from_start = np.arange(traj.shape[0]) * dt
    return traj, time_from_start


def traj_tour_from_lininterp_qdot(Q, qdot):
    """
    Creating tour trajectory given the sequence of waypoints configurations Q.
    Each segment between two waypoints is interpolated based on the joint velocities qdot.
    This ensures that the time taken for each segment is proportional to the distance and joint velocities.

    Q: (n, dof)
    qdot: (dof,) array of joint velocities
    return: (m, dof) where m is the total number of interpolated points
    """
    traj = []
    time_from_start = []
    for i in range(Q.shape[0] - 1):
        # Compute the distance for each joint
        dist = np.abs(Q[i + 1] - Q[i])
        # Compute the time required for each joint based on qdot
        time_required = dist / qdot
        # The maximum time required across all joints determines the number of interpolation points
        max_time = np.max(time_required)
        num_points = int(np.ceil(max_time * 100))  # Assuming 100 Hz sampling rate
        if num_points < 2:
            num_points = 2  # Ensure at least two points for interpolation
        # Interpolate between Q[i] and Q[i + 1]
        interp_segment = np.linspace(Q[i], Q[i + 1], num_points)
        traj.append(interp_segment)
        # Compute time_from_start for this segment
        if len(time_from_start) == 0:
            time_from_start.append(np.linspace(0, max_time, num_points))
        else:
            last_time = time_from_start[-1][-1]
            time_from_start.append(
                np.linspace(last_time, last_time + max_time, num_points)
            )
    traj = np.vstack(traj)
    time_from_start = np.hstack(time_from_start)
    return traj, time_from_start


# def traj_tour_from_collisionfree_planner(Q, qdot):
