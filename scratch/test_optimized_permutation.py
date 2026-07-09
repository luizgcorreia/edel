import numpy as np
import time
import ot
from scipy.spatial.distance import cdist

def compute_wasserstein(X: np.ndarray, Y: np.ndarray) -> float:
    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = cdist(X, Y, metric="euclidean")
    try:
        return float(ot.emd2(a, b, M))
    except Exception:
        return float(np.mean(M))

def compute_h2_original(X: np.ndarray, Y: np.ndarray, k=10, B=100, max_queries=50):
    N = X.shape[0]
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    query_indices = np.random.choice(N, size=min(max_queries, N), replace=False)
    
    obs_w_dists = []
    for j in query_indices:
        neighbors = np.argsort(sim[j])[-k:]
        Y_neighbors = Y[neighbors]
        random_idx = np.random.choice(N, size=k, replace=False)
        Y_random = Y[random_idx]
        obs_w_dists.append(compute_wasserstein(Y_neighbors, Y_random))
    W_obs = float(np.mean(obs_w_dists))

    perm_W_vals = []
    for _ in range(B):
        perm_idx = np.random.permutation(N)
        Y_perm = Y[perm_idx]
        perm_w_dists = []
        for j in query_indices:
            neighbors = np.argsort(sim[j])[-k:]
            Y_neighbors = Y_perm[neighbors]
            random_idx = np.random.choice(N, size=k, replace=False)
            Y_random = Y_perm[random_idx]
            perm_w_dists.append(compute_wasserstein(Y_neighbors, Y_random))
        perm_W_vals.append(np.mean(perm_w_dists))
        
    p_val = float((1.0 + np.sum(np.array(perm_W_vals) >= W_obs)) / (1.0 + B))
    return W_obs, p_val

def compute_h2_optimized(X: np.ndarray, Y: np.ndarray, k=10, B=1000, max_queries=50, pool_size=500):
    N = X.shape[0]
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    query_indices = np.random.choice(N, size=min(max_queries, N), replace=False)
    
    obs_w_dists = []
    for j in query_indices:
        neighbors = np.argsort(sim[j])[-k:]
        Y_neighbors = Y[neighbors]
        random_idx = np.random.choice(N, size=k, replace=False)
        Y_random = Y[random_idx]
        obs_w_dists.append(compute_wasserstein(Y_neighbors, Y_random))
    W_obs = float(np.mean(obs_w_dists))

    # Pre-generate a pool of random Wasserstein distances
    # of size pool_size (e.g. 500)
    pool_distances = []
    for _ in range(pool_size):
        idx1 = np.random.choice(N, size=k, replace=False)
        idx2 = np.random.choice(N, size=k, replace=False)
        pool_distances.append(compute_wasserstein(Y[idx1], Y[idx2]))
    pool_distances = np.array(pool_distances)

    # Bootstrap B times by drawing max_queries samples with replacement from the pool
    # and taking their mean
    # We can vectorize this draw!
    draws = np.random.choice(pool_distances, size=(B, min(max_queries, N)))
    perm_W_vals = np.mean(draws, axis=1)
    
    p_val = float((1.0 + np.sum(perm_W_vals >= W_obs)) / (1.0 + B))
    return W_obs, p_val

# Test with synthetic data
np.random.seed(42)
N, dims = 300, 16
X = np.random.randn(N, dims)
Y = X + np.random.randn(N, dims) * 0.1 # organized

print("Running Original H2 (B=100)...")
t0 = time.time()
w_obs_orig, p_val_orig = compute_h2_original(X, Y, B=100)
t_orig = time.time() - t0
print(f"Original: W_obs={w_obs_orig:.4f}, p-val={p_val_orig:.4f}, Time={t_orig:.4f}s")

print("\nRunning Optimized H2 (B=1000, pool_size=500)...")
t0 = time.time()
w_obs_opt, p_val_opt = compute_h2_optimized(X, Y, B=1000, pool_size=500)
t_opt = time.time() - t0
print(f"Optimized: W_obs={w_obs_opt:.4f}, p-val={p_val_opt:.4f}, Time={t_opt:.4f}s")
