import numpy as np
from paper_sequential_planner.scripts.geometric_poses import (
    poses_epGH,
    se3_error_pairwise_distance,
)

H = poses_epGH()
print(f"==>> H.shape: \n{H.shape}")
tspace_dist = se3_error_pairwise_distance(H, 0.2)
print(f"==>> tspace_dist.shape: \n{tspace_dist.shape}")


def radius_neighbors(D, radius):
    neighbors = []
    for i in range(D.shape[0]):
        idx = np.where(D[i] < radius)[0]
        idx = idx[idx != i]  # remove self
        neighbors.append(idx.tolist())
    return neighbors


def knn_from_distance(D, k=5):
    # ignore self-distance by setting diagonal large
    D = D.copy()
    np.fill_diagonal(D, np.inf)

    idx = np.argpartition(D, k, axis=1)[:, :k]  # (N, k)

    # optional: sort neighbors by distance
    row_idx = np.arange(D.shape[0])[:, None]
    sorted_order = np.argsort(D[row_idx, idx], axis=1)
    idx = idx[row_idx, sorted_order]

    return idx.tolist()  # indices of k nearest per row


def task_space_correlation():
    nnr = 0.15
    nnk = 5
    nn_r = radius_neighbors(tspace_dist, radius=nnr)
    nn_k = knn_from_distance(tspace_dist, k=nnk)
    nn_union = []
    for i in range(tspace_dist.shape[0]):
        union_set = set(nn_r[i]) | set(nn_k[i])
        nn_union.append(sorted(union_set))
    nn_dist = []
    for i in range(len(nn_union)):
        dists = [tspace_dist[i, j].item() for j in nn_union[i]]
        nn_dist.append(dists)

    nn_count = [len(n) for n in nn_union]
    return nn_union, nn_dist, nn_count, nn_r, nn_k


nn_union, nn_dist, nn_count, nn_r, nn_k = task_space_correlation()
print(f"==>> nn_count: \n{nn_count}")

sum_count = sum(nn_count)
print(f"==>> sum_count: \n{sum_count}")
