import numpy as np
import time
import os

# Vectorized Fastron helpers


def compute_kernel_gram_matrix_vectorized(data, gamma, dtype=None):
    """Compute the full RBF (Gaussian) Gram matrix in a vectorized way.

    G[i,j] = exp(-gamma * ||data[i] - data[j]||^2)

    Parameters
    - data: (N, d) array
    - gamma: scalar
    - dtype: if provided, cast arrays to this dtype (e.g., np.float32)

    Returns
    - G: (N, N) Gram matrix
    """
    if dtype is not None:
        data = data.astype(dtype, copy=False)

    # squared norms (N,)
    sq_norms = np.sum(data * data, axis=1)
    # pairwise squared distances using the identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    pairwise_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * data.dot(data.T)
    # numerical safety
    np.maximum(pairwise_sq, 0.0, out=pairwise_sq)
    G = np.exp(-gamma * pairwise_sq)
    return G


def kernel_vector_to_query(data, queryPoint, gamma):
    """Compute vector of RBF kernel values between each row in data and queryPoint.

    Returns k of shape (N,)
    """
    query = np.asarray(queryPoint)
    diffs = data - query
    sq_dists = np.sum(diffs * diffs, axis=1)
    return np.exp(-gamma * sq_dists)


def hypothesisf_vectorized(queryPoint, data, alpha, gamma):
    """Vectorized hypothesis: sign(alpha . k(queryPoint))"""
    k = kernel_vector_to_query(data, queryPoint, gamma)
    score = alpha.dot(k)
    return np.sign(score)


def eval_vectorized(queryPoint, data, alpha, gamma):
    """Same as hypothesisf_vectorized, keeps the name `eval` used in original script."""
    return hypothesisf_vectorized(queryPoint, data, alpha, gamma)


def F_from_alpha(G, alpha):
    """Compute F = G.dot(alpha) (raw scores)."""
    return G.dot(alpha)


# --- small demo / benchmark comparing to a naive implementation ---

def naive_compute_kernel_gram_matrix(data, gamma):
    N = data.shape[0]
    G = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            d = np.linalg.norm(data[i] - data[j])
            G[i, j] = np.exp(-gamma * d * d)
    return G


def approx_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol, rtol=1e-6)


if __name__ == "__main__":
    # Try to reuse the same dataset as the original script if RSRC_DIR is set.
    rsrc = os.environ.get("RSRC_DIR")
    if rsrc:
        try:
            dataset = np.load(os.path.join(rsrc, "cspace_dataset.npy"))
            print("Loaded dataset from RSRC_DIR")
            trainsize = 600
            rng = np.random.default_rng(42)
            samples_id = rng.choice(range(dataset.shape[0]), size=trainsize, replace=False)
            samples = dataset[samples_id]
            data = samples[:, 0:2].astype(np.float64)
            y = samples[:, 2]
        except Exception as e:
            print("Failed to load dataset, falling back to synthetic. Error:", e)
            rsrc = None

    if not rsrc:
        # synthetic small dataset for demo
        rng = np.random.default_rng(42)
        N = 800
        data = rng.normal(size=(N, 2)).astype(np.float64)
        y = rng.choice([-1, 1], size=(N,))

    gamma = 10.0

    print(f"Dataset shape: {data.shape}, gamma={gamma}")

    # Warm up / small correctness check
    print("Computing Gram matrix (vectorized)...")
    t0 = time.perf_counter()
    G_vec = compute_kernel_gram_matrix_vectorized(data, gamma)
    t1 = time.perf_counter()
    print(f"Vectorized G computed in {t1-t0:.4f}s")

    # Compute naive (only if N is reasonably small) to verify correctness
    N = data.shape[0]
    if N <= 1500:
        print("Computing Gram matrix (naive) for correctness check...")
        t0 = time.perf_counter()
        G_naive = naive_compute_kernel_gram_matrix(data, gamma)
        t1 = time.perf_counter()
        print(f"Naive G computed in {t1-t0:.4f}s")
        print("Compare G matrices:", "OK" if approx_equal(G_vec, G_naive, tol=1e-8) else "DIFFER")

    # Demonstrate hypothesis / eval speed
    rng = np.random.default_rng(1)
    alpha = rng.random(size=(data.shape[0],)) - 0.5

    q = np.array([1.0, 1.0])
    # naive hypothesis (loop-based) for comparison
    def hypothesis_naive(queryPoint, data, alpha, gamma):
        term = []
        for i, xi in enumerate(data):
            d = np.linalg.norm(xi - queryPoint)
            term.append(alpha[i] * np.exp(-gamma * d * d))
        return np.sign(sum(term))

    # time naive
    t0 = time.perf_counter()
    s_naive = hypothesis_naive(q, data, alpha, gamma)
    t1 = time.perf_counter()
    naive_time = t1 - t0

    # time vectorized
    t0 = time.perf_counter()
    s_vec = hypothesisf_vectorized(q, data, alpha, gamma)
    t1 = time.perf_counter()
    vec_time = t1 - t0

    print(f"hypothesis naive: {s_naive} in {naive_time:.6f}s")
    print(f"hypothesis vectorized: {s_vec} in {vec_time:.6f}s")
    if naive_time > 0:
        print(f"Speedup: {naive_time/vec_time:.2f}x")

    # Compute F via matrix multiply and via repeated hypothesis calls (slow)
    t0 = time.perf_counter()
    F_vec = F_from_alpha(G_vec, alpha)
    t1 = time.perf_counter()
    print(f"F via G.dot(alpha) computed in {t1-t0:.6f}s")

    # approximate F via many vectorized kernel computations (batch)
    t0 = time.perf_counter()
    # build kernel matrix by computing kernels to each data point (this is the same as G)
    # but we demonstrate the idiom used in some non-vectorized code: many kernel evaluations
    K_rows = [kernel_vector_to_query(data, data[i], gamma) for i in range(min(200, N))]
    t1 = time.perf_counter()
    print(f"Computed {len(K_rows)} kernel-to-point vectors in {t1-t0:.4f}s (demonstration only)")

    print("Done. Replace nested loops in your original file with these functions for large speedups.")

    numsamples = 360
    theta1 = np.linspace(-np.pi, np.pi, numsamples)
    theta2 = np.linspace(-np.pi, np.pi, numsamples)

    cspace_points = []
    T1, T2 = np.meshgrid(theta1, theta2)
    for i in range(numsamples):
        for j in range(numsamples):
            theta = np.array([T1[i, j], T2[i, j]])
            collision = eval_vectorized(theta, data, alpha, gamma)
            if collision == 1:
                cspace_points.append(theta)
    cspace_points = np.array(cspace_points)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(
        cspace_points[:, 0],
        cspace_points[:, 1],
        "ro",
        markersize=1,
        label="C-space Obstacles",
    )
    ax.set_xlabel("Theta 1")
    ax.set_ylabel("Theta 2")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", "box")
    ax.set_title("Fastron C-space Obstacle Approximation (Vectorized)")
    ax.legend()
    plt.show()
