import numpy as np
import time
from sklearn.linear_model import Ridge
import scipy.spatial.distance as dist
import ot

def compute_wasserstein(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0:
        return 0.0
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = dist.cdist(X, Y, metric="euclidean")
    try:
        val = ot.emd2(a, b, M)
        return float(val)
    except Exception:
        return float(np.mean(M))

def benchmark():
    N = 800
    dims = 1536 # True embedding dimensions
    np.random.seed(42)
    emb_p = np.random.randn(N, dims)
    emb_i = np.random.randn(N, dims)

    split_idx = int(0.7 * N)
    hist_mask = np.zeros(N, dtype=bool)
    hist_mask[:split_idx] = True
    fut_mask = ~hist_mask

    # Option 1: Temporal Permutation (10 iterations)
    t0 = time.time()
    for _ in range(10):
        # Shuffle mask
        shuf_indices = np.random.permutation(N)
        shuf_hist_mask = np.zeros(N, dtype=bool)
        shuf_hist_mask[shuf_indices[:split_idx]] = True
        shuf_fut_mask = ~shuf_hist_mask

        I_hist = emb_i[shuf_hist_mask]
        P_hist = emb_p[shuf_hist_mask]
        I_fut = emb_i[shuf_fut_mask]
        P_fut = emb_p[shuf_fut_mask]

        # NumPy Dual formulation of Ridge (closed-form)
        alpha = 1.0
        n_h = len(I_hist)
        K = I_hist @ I_hist.T
        # Solve (K + alpha*I) A = P_hist
        A = np.linalg.solve(K + alpha * np.eye(n_h), P_hist)
        # Predict: P_pred = I_fut @ I_hist.T @ A
        P_pred = (I_fut @ I_hist.T) @ A

        w_edel = compute_wasserstein(P_pred, P_fut)
        w_baseline = compute_wasserstein(P_hist, P_fut)
        gain = w_baseline - w_edel
    t1 = time.time()
    print(f"Option 1 (NumPy Dual Permutation) 10 iterations: {t1 - t0:.4f}s (avg: {(t1 - t0)/10:.4f}s)")

    # Option 2: Future Bootstrap (10 iterations)
    # Train model once on true history
    I_hist_true = emb_i[hist_mask]
    P_hist_true = emb_p[hist_mask]
    I_fut_true = emb_i[fut_mask]
    P_fut_true = emb_p[fut_mask]
    reg_true = Ridge(alpha=1.0)
    reg_true.fit(I_hist_true, P_hist_true)
    P_pred_true = reg_true.predict(I_fut_true)

    fut_indices = np.arange(len(P_fut_true))

    t2 = time.time()
    for _ in range(10):
        boot_idx = np.random.choice(fut_indices, size=len(fut_indices), replace=True)
        P_fut_b = P_fut_true[boot_idx]
        P_pred_b = P_pred_true[boot_idx]

        w_edel = compute_wasserstein(P_pred_b, P_fut_b)
        w_baseline = compute_wasserstein(P_hist_true, P_fut_b)
        gain = w_baseline - w_edel
    t3 = time.time()
    print(f"Option 2 (Future Bootstrap) 10 iterations: {t3 - t2:.4f}s (avg: {(t3 - t2)/10:.4f}s)")

if __name__ == "__main__":
    benchmark()
