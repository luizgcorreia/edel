"""Hypothesis testing metrics plugin module for EDEL.

Implements:
- H1: Structural transition tests (observed vs aspect-shuffled KS test).
- H2: Local transition organization (Wasserstein distance permutation test).
- H3: Predictive capacity (temporal split prediction Wasserstein vs baseline, and Bivariate Moran's I spatial alignment significance).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from edel.pipeline.projection import load_embeddings_to_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wasserstein Distance Helper
# ---------------------------------------------------------------------------

def compute_wasserstein(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the 1st Wasserstein distance (Earth Mover's Distance) using POT."""
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
        # Fallback to mean Euclidean distance if linear solver fails
        return float(np.mean(M))


# ---------------------------------------------------------------------------
# H2 Local Transition Organization helper
# ---------------------------------------------------------------------------

def compute_h2_for_transition(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10,
    B: int = 20,
    max_queries: int = 50,
) -> tuple[float, float]:
    """Compute observed neighborhood Wasserstein distance and permuted significance.

    Lower Wasserstein distance compared to random target sets indicates local clustering/organization.
    """
    N = X.shape[0]
    if N < k + 1:
        return 0.0, 1.0

    # Distance matrix in X space (use cosine since they are normalized)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)

    # Subsample query points to keep computation fast
    query_indices = np.random.choice(N, size=min(max_queries, N), replace=False)

    obs_w_dists = []
    for j in query_indices:
        neighbors = np.argsort(sim[j])[-k:]
        Y_neighbors = Y[neighbors]
        # Null model: random target points from Y
        random_idx = np.random.choice(N, size=k, replace=False)
        Y_random = Y[random_idx]
        obs_w_dists.append(compute_wasserstein(Y_neighbors, Y_random))

    W_obs = float(np.mean(obs_w_dists))

    # Permutation test
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

    # Empirical p-value: fraction of permutations where Wasserstein distance is smaller/equal to observed
    p_val = float((1.0 + np.sum(np.array(perm_W_vals) <= W_obs)) / (1.0 + B))
    return W_obs, p_val


# ---------------------------------------------------------------------------
# Bivariate Moran's I helper
# ---------------------------------------------------------------------------

def compute_morans_i(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Compute Bivariate Moran's I statistic for two spatial variables."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_x = x - x_mean
    z_y = y - y_mean

    numerator = 0.0
    for i in range(n):
        for j in range(n):
            numerator += w[i, j] * z_x[i] * z_y[j]

    denominator = np.sum(z_x ** 2)
    if denominator == 0:
        return 0.0

    W_sum = np.sum(w)
    if W_sum == 0:
        return 0.0

    I_val = (n / W_sum) * (numerator / denominator)
    return float(I_val)


# ---------------------------------------------------------------------------
# Main Hypothesis Metrics Plugin
# ---------------------------------------------------------------------------

def hypothesis_metrics(artifacts: dict) -> dict:
    """Run H1, H2, and H3 tests and record metrics and feature outputs."""
    df: pd.DataFrame = artifacts.get("embedding")
    if df is None:
        return {"metrics": {}, "features": {}}

    dimensions = artifacts.get("_dimensions", 1536)
    aspects = ["problem", "method", "finding", "interpretation"]
    if not all(f"{a}_embedding" in df.columns for a in aspects):
        return {"metrics": {}, "features": {}}

    # Load and normalize matrices
    def load(aspect: str) -> np.ndarray:
        mat = load_embeddings_to_matrix(df, f"{aspect}_embedding", dimensions)
        mat -= mat.mean(axis=0)
        return sk_normalize(mat)

    emb_p = load("problem")
    emb_m = load("method")
    emb_f = load("finding")
    emb_i = load("interpretation")

    metrics: dict = {}
    features: dict = {}

    # -----------------------------------------------------------------------
    # H1 Test: Structural Transition
    # -----------------------------------------------------------------------
    N = emb_p.shape[0]
    shuffled_p = emb_p[np.random.permutation(N)]
    shuffled_m = emb_m[np.random.permutation(N)]
    shuffled_f = emb_f[np.random.permutation(N)]
    shuffled_i = emb_i[np.random.permutation(N)]

    # Operators
    pm = emb_m - emb_p
    mf = emb_f - emb_m
    fi = emb_i - emb_f
    pm_s = shuffled_m - shuffled_p
    mf_s = shuffled_f - shuffled_m
    fi_s = shuffled_i - shuffled_f

    # Norms
    norm_pm = np.linalg.norm(pm, axis=1)
    norm_mf = np.linalg.norm(mf, axis=1)
    norm_fi = np.linalg.norm(fi, axis=1)
    norm_pm_s = np.linalg.norm(pm_s, axis=1)
    norm_mf_s = np.linalg.norm(mf_s, axis=1)
    norm_fi_s = np.linalg.norm(fi_s, axis=1)

    # Cosines
    def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(sk_normalize(a) * sk_normalize(b), axis=1)

    cos_pm_mf = row_cos(pm, mf)
    cos_pm_fi = row_cos(pm, fi)
    cos_mf_fi = row_cos(mf, fi)
    cos_pm_mf_s = row_cos(pm_s, mf_s)
    cos_pm_fi_s = row_cos(pm_s, fi_s)
    cos_mf_fi_s = row_cos(mf_s, fi_s)

    # Compute KS tests
    def run_ks(obs, shuf, prefix):
        res = ks_2samp(obs, shuf)
        metrics[f"h1_ks_stat_{prefix}"] = float(res.statistic)
        metrics[f"h1_ks_pvalue_{prefix}"] = float(res.pvalue)

    run_ks(norm_pm, norm_pm_s, "norm_pm")
    run_ks(norm_mf, norm_mf_s, "norm_mf")
    run_ks(norm_fi, norm_fi_s, "norm_fi")
    run_ks(cos_pm_mf, cos_pm_mf_s, "cos_pm_mf")
    run_ks(cos_pm_fi, cos_pm_fi_s, "cos_pm_fi")
    run_ks(cos_mf_fi, cos_mf_fi_s, "cos_mf_fi")

    # -----------------------------------------------------------------------
    # H2 Test: Local Transition Organization
    # -----------------------------------------------------------------------
    # Test for P -> M, M -> F, F -> I
    w_pm, p_pm = compute_h2_for_transition(emb_p, emb_m)
    w_mf, p_mf = compute_h2_for_transition(emb_m, emb_f)
    w_fi, p_fi = compute_h2_for_transition(emb_f, emb_i)

    metrics["h2_w_dist_pm"] = w_pm
    metrics["h2_pvalue_pm"] = p_pm
    metrics["h2_w_dist_mf"] = w_mf
    metrics["h2_pvalue_mf"] = p_mf
    metrics["h2_w_dist_fi"] = w_fi
    metrics["h2_pvalue_fi"] = p_fi

    # -----------------------------------------------------------------------
    # H3 Test: Predictive Transition Capacity
    # -----------------------------------------------------------------------
    # Time split logic
    years = df.get("publication_year")
    valid_years = []
    if years is not None:
        for y in years.values:
            try:
                valid_years.append(int(float(y)))
            except:
                valid_years.append(-1)
    valid_years = np.array(valid_years)

    unique_years = np.unique(valid_years[valid_years > 0])

    if len(unique_years) >= 2:
        split_year = np.percentile(valid_years[valid_years > 0], 70)
        hist_mask = (valid_years > 0) & (valid_years <= split_year)
        fut_mask = (valid_years > 0) & (valid_years > split_year)
    else:
        split_idx = int(0.7 * N)
        hist_mask = np.zeros(N, dtype=bool)
        hist_mask[:split_idx] = True
        fut_mask = ~hist_mask

    # Handle tiny splits
    if hist_mask.sum() < 5 or fut_mask.sum() < 5:
        split_idx = N // 2
        hist_mask = np.zeros(N, dtype=bool)
        hist_mask[:split_idx] = True
        fut_mask = ~hist_mask

    I_hist = emb_i[hist_mask]
    P_hist = emb_p[hist_mask]
    I_fut = emb_i[fut_mask]
    P_fut = emb_p[fut_mask]

    # Predict problem using Ridge transition operator trained on historical set
    reg = Ridge(alpha=1.0)
    reg.fit(I_hist, P_hist)
    P_pred = reg.predict(I_fut)

    # Global Wasserstein evaluation
    w_edel = compute_wasserstein(P_pred, P_fut)
    w_baseline = compute_wasserstein(P_hist, P_fut)

    metrics["h3_w_edel"] = w_edel
    metrics["h3_w_baseline"] = w_baseline
    metrics["h3_predictive_gain"] = float(w_baseline - w_edel)

    # Spatial predictive alignment using KMeans and Bivariate Moran's I
    n_clusters = min(10, N)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(emb_p)
    centroids = kmeans.cluster_centers_

    # Counts
    hist_labels = kmeans.predict(P_hist)
    fut_labels = kmeans.predict(P_fut)
    pred_labels = kmeans.predict(P_pred)

    c_hist = np.bincount(hist_labels, minlength=n_clusters)
    c_fut = np.bincount(fut_labels, minlength=n_clusters)
    c_pred = np.bincount(pred_labels, minlength=n_clusters)

    # Densities changes
    x = c_pred - c_hist
    y = c_fut - c_hist

    # Centroid distances
    d_ij = cdist(centroids, centroids, metric="euclidean")
    with np.errstate(divide="ignore"):
        w = 1.0 / d_ij
    w[np.isinf(w)] = 0.0
    np.fill_diagonal(w, 0.0)

    # Row-standardize weight matrix
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w = w / row_sums

    I_obs = compute_morans_i(x, y, w)
    metrics["h3_moran_i"] = I_obs

    # Moran's I permutation significance
    B_moran = 50
    moran_null_vals = []
    for _ in range(B_moran):
        shuf_indices = np.random.permutation(N)
        shuf_hist_mask = np.zeros(N, dtype=bool)
        shuf_hist_mask[shuf_indices[:hist_mask.sum()]] = True
        shuf_fut_mask = ~shuf_hist_mask

        I_hist_s = emb_i[shuf_hist_mask]
        P_hist_s = emb_p[shuf_hist_mask]
        I_fut_s = emb_i[shuf_fut_mask]
        P_fut_s = emb_p[shuf_fut_mask]

        reg_s = Ridge(alpha=1.0)
        reg_s.fit(I_hist_s, P_hist_s)
        P_pred_s = reg_s.predict(I_fut_s)

        hist_labels_s = kmeans.predict(P_hist_s)
        fut_labels_s = kmeans.predict(P_fut_s)
        pred_labels_s = kmeans.predict(P_pred_s)

        c_hist_s = np.bincount(hist_labels_s, minlength=n_clusters)
        c_fut_s = np.bincount(fut_labels_s, minlength=n_clusters)
        c_pred_s = np.bincount(pred_labels_s, minlength=n_clusters)

        x_s = c_pred_s - c_hist_s
        y_s = c_fut_s - c_hist_s

        moran_null_vals.append(compute_morans_i(x_s, y_s, w))

    p_moran = float((1.0 + np.sum(np.array(moran_null_vals) >= I_obs)) / (1.0 + B_moran))
    metrics["h3_moran_pvalue"] = p_moran

    # Features (store raw distributions for custom comparisons)
    features["h1_shuffled_norms"] = {
        "norm_pm": norm_pm_s.astype(np.float32),
        "norm_mf": norm_mf_s.astype(np.float32),
        "norm_fi": norm_fi_s.astype(np.float32),
    }
    features["h1_shuffled_cosines"] = {
        "cos_pm_mf": cos_pm_mf_s.astype(np.float32),
        "cos_pm_fi": cos_pm_fi_s.astype(np.float32),
        "cos_mf_fi": cos_mf_fi_s.astype(np.float32),
    }

    return {"metrics": metrics, "features": features}
