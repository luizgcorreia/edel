import numpy as np
import ot
from scipy.spatial.distance import cdist

def compute_wasserstein(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0:
        return 0.0
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = cdist(X, Y, metric="euclidean")
    try:
        val = ot.emd2(a, b, M)
        return float(val)
    except Exception:
        return float(np.mean(M))

def analyze_distributions(X, Y, label, k=10, B=20, max_queries=50):
    N = X.shape[0]
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    query_indices = np.random.choice(N, size=min(max_queries, N), replace=False)
    
    obs_w_dists = []
    for j in query_indices:
        neighbors = np.argsort(sim[j])[-k:]
        Y_neighbors = Y[neighbors]
        Y_random = Y[np.random.choice(N, size=k, replace=False)]
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
            Y_random = Y_perm[np.random.choice(N, size=k, replace=False)]
            perm_w_dists.append(compute_wasserstein(Y_neighbors, Y_random))
        perm_W_vals.append(np.mean(perm_w_dists))

    perm_W_vals = np.array(perm_W_vals)
    print(f"\n--- Analysis for {label} ---")
    print(f"W_obs: {W_obs:.4f}")
    print(f"perm_W range: [{perm_W_vals.min():.4f}, {perm_W_vals.max():.4f}], mean: {perm_W_vals.mean():.4f}")
    p_orig = float((1.0 + np.sum(perm_W_vals <= W_obs)) / (1.0 + B))
    p_corr = float((1.0 + np.sum(perm_W_vals >= W_obs)) / (1.0 + B))
    print(f"Original p-val (<=): {p_orig:.4f}")
    print(f"Corrected p-val (>=): {p_corr:.4f}")

np.random.seed(42)
N = 100
d = 16
X = np.random.normal(0, 1, size=(N, d))
X /= np.linalg.norm(X, axis=1, keepdims=True)

# Structured Y
beta = np.random.normal(0, 1, size=(d, d))
Y = X @ beta + np.random.normal(0, 0.1, size=(N, d))
Y /= np.linalg.norm(Y, axis=1, keepdims=True)
analyze_distributions(X, Y, "Structured Data", B=100)

# Unstructured Y
Y_rand = np.random.normal(0, 1, size=(N, d))
Y_rand /= np.linalg.norm(Y_rand, axis=1, keepdims=True)
analyze_distributions(X, Y_rand, "Unstructured Data", B=100)
