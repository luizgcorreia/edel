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

# Let's generate a global distribution of Y in d-dimensional space (e.g., d=10)
np.random.seed(42)
N = 1000
d = 10
Y_global = np.random.normal(0, 1, size=(N, d))
Y_global /= np.linalg.norm(Y_global, axis=1, keepdims=True) # project to sphere like normalized embeddings

k = 10

# Case 1: No local organization (Y_neighbors is just a random sample from Y_global)
w_dists_null = []
for _ in range(100):
    Y_neighbors = Y_global[np.random.choice(N, k, replace=False)]
    Y_random = Y_global[np.random.choice(N, k, replace=False)]
    w_dists_null.append(compute_wasserstein(Y_neighbors, Y_random))

# Case 2: Local organization (Y_neighbors is a tight cluster)
# We pick a center and generate points close to it, then project to sphere
w_dists_organized = []
for _ in range(100):
    center = Y_global[np.random.choice(N)]
    # a tight cluster around center
    Y_neighbors = center + np.random.normal(0, 0.05, size=(k, d))
    Y_neighbors /= np.linalg.norm(Y_neighbors, axis=1, keepdims=True)
    Y_random = Y_global[np.random.choice(N, k, replace=False)]
    w_dists_organized.append(compute_wasserstein(Y_neighbors, Y_random))

print(f"Mean Wasserstein (two random samples): {np.mean(w_dists_null):.4f}")
print(f"Mean Wasserstein (tight cluster vs random): {np.mean(w_dists_organized):.4f}")
