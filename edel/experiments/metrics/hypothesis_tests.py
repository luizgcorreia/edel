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
from scipy.stats import ks_2samp, wasserstein_distance as scipy_wasserstein_distance
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from edel.pipeline.projection import load_embeddings_to_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wasserstein Distance Helper
# ---------------------------------------------------------------------------

def compute_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    max_samples: int = 2000,
    idx_X: np.ndarray | None = None,
    idx_Y: np.ndarray | None = None,
) -> float:
    """Compute the 1st Wasserstein distance (Earth Mover's Distance) using POT.

    Parameters
    ----------
    idx_X, idx_Y : optional fixed subsample indices.  When provided, these
        override the internal random subsample and guarantee that the *same*
        rows are used across multiple calls (e.g. observed + permutations in
        H3), eliminating subsampling variance from the p-value.
    """
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0:
        return 0.0

    # Subsample X
    if idx_X is not None:
        X = X[idx_X]
    elif n > max_samples:
        rng = np.random.RandomState(42)
        idx_X = rng.choice(n, size=max_samples, replace=False)
        X = X[idx_X]
    # Subsample Y
    if idx_Y is not None:
        Y = Y[idx_Y]
    elif m > max_samples:
        rng = np.random.RandomState(42)
        idx_Y = rng.choice(m, size=max_samples, replace=False)
        Y = Y[idx_Y]

    n, m = X.shape[0], Y.shape[0]

    a = np.ones(n) / n
    b = np.ones(m) / m
    M = cdist(X, Y, metric="euclidean")

    try:
        val = ot.emd2(a, b, M)
        return float(val)
    except Exception:
        # Fallback to mean Euclidean distance if linear solver fails
        return float(np.mean(M))


def compute_wasserstein_sliced(
    X: np.ndarray, Y: np.ndarray, n_projections: int = 200
) -> float:
    """Sliced 1-Wasserstein distance using random projections (no sample cap).

    Uses all available data (no max_samples cap), making it suitable for
    large-scale permutation tests where exact EMD would require subsampling.
    The projection approximation bias cancels out in permutation comparisons
    since both observed and null gains use the same projections.
    """
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0:
        return 0.0
    try:
        return float(ot.sliced_wasserstein_distance(X, Y, n_projections=n_projections, seed=42, p=1))
    except Exception:
        return float(np.mean(cdist(X, Y, metric="euclidean")))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Squared energy distance D²(X,Y) for multivariate two-sample testing.

    D² = 2·E‖x−y‖ − E‖x−x'‖ − E‖y−y'‖

    A value of 0 means the two distributions are identical; larger values
    indicate increasing dissimilarity.  Distribution-free and consistent
    against all alternatives.
    """
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return 0.0
    XX = float(np.mean(cdist(X, X, metric="euclidean")))
    YY = float(np.mean(cdist(Y, Y, metric="euclidean")))
    XY = float(np.mean(cdist(X, Y, metric="euclidean")))
    return float(2.0 * XY - XX - YY)


# ---------------------------------------------------------------------------
# H2 Local Transition Organization helper
# ---------------------------------------------------------------------------

def compute_h2_for_transition(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10,
    B: int = 1000,
    max_queries: int = 50,
) -> tuple[float, float, float]:
    """Compute observed neighborhood Wasserstein distance, permuted significance, and z-score.

    Returns:
        W_obs:  Mean observed neighborhood Wasserstein distance.
        p_val:  Empirical p-value from bootstrap permutation test.
        z_score: Effect size z = (W_obs - mu_rand) / sigma_rand.
    """
    N = X.shape[0]
    if N < k + 1:
        return 0.0, 1.0, 0.0

    # Subsample query points to keep computation fast
    query_indices = np.random.choice(N, size=min(max_queries, N), replace=False)

    # Compute similarity matrix only for query points to avoid N x N allocation/multiplication
    sim_queries = X[query_indices] @ X.T

    obs_w_dists = []
    for idx, j in enumerate(query_indices):
        sim_row = sim_queries[idx].copy()
        sim_row[j] = -np.inf  # avoid self-neighbor
        neighbors = np.argsort(sim_row)[-k:]
        Y_neighbors = Y[neighbors]
        # Null model: random target points from Y
        random_idx = np.random.choice(N, size=k, replace=False)
        Y_random = Y[random_idx]
        obs_w_dists.append(compute_wasserstein(Y_neighbors, Y_random))

    W_obs = float(np.mean(obs_w_dists))

    # Pre-generate a pool of random Wasserstein distances under the null hypothesis.
    # Under the null, both neighbor and random target sets are random samples of size k from Y.
    pool_size = 500
    pool_distances = []
    for _ in range(pool_size):
        idx1 = np.random.choice(N, size=k, replace=False)
        idx2 = np.random.choice(N, size=k, replace=False)
        pool_distances.append(compute_wasserstein(Y[idx1], Y[idx2]))
    pool_distances = np.array(pool_distances)

    # Bootstrap B times by drawing max_queries samples with replacement from the pool
    # and taking their mean
    draws = np.random.choice(pool_distances, size=(B, len(query_indices)))
    perm_W_vals = np.mean(draws, axis=1)

    # Empirical p-value: fraction of permutations where Wasserstein distance is >= observed
    p_val = float((1.0 + np.sum(perm_W_vals >= W_obs)) / (1.0 + B))

    # Effect size: z-score relative to the null distribution
    mu_rand = float(np.mean(perm_W_vals))
    sigma_rand = float(np.std(perm_W_vals))
    z_score = float((W_obs - mu_rand) / sigma_rand) if sigma_rand > 0 else 0.0

    return W_obs, p_val, z_score


def compute_h2b_for_pair(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10,
    B: int = 200,
) -> dict:
    """Compute local transition asymmetry metrics and permutation significance.

    Calculates:
      - Forward average entropy and effective branching factor (X -> Y).
      - Reverse average entropy and effective branching factor (Y -> X).
      - Entropy difference (Forward - Reverse).
      - Permutation p-value under null hypothesis of symmetry (by randomly swapping X and Y directions).
    """
    N = X.shape[0]
    if N < k + 1:
        return {
            "entropy_forward": 0.0,
            "entropy_reverse": 0.0,
            "branching_forward": 1.0,
            "branching_reverse": 1.0,
            "diff": 0.0,
            "pvalue": 1.0,
        }

    # Sub-sample queries to prevent performance bottlenecks on large datasets
    max_queries = 300
    if N > max_queries:
        # Fix the seed for reproducibility of sub-sampling within the function call
        rng = np.random.default_rng(42)
        query_indices = rng.choice(N, size=max_queries, replace=False)
    else:
        query_indices = np.arange(N)

    epsilon = 1e-5
    xy_stacked = np.stack([X, Y], axis=0) # Shape: (2, N, d)

    # Precompute dot products for fast vectorized neighbor calculations
    dot_XX = X[query_indices] @ X.T
    dot_XY = X[query_indices] @ Y.T
    dot_YX = Y[query_indices] @ X.T
    dot_YY = Y[query_indices] @ Y.T

    # Helper function for observed/non-permuted metrics
    def compute_directed_metrics(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, float, float]:
        sim = src[query_indices] @ src.T
        sim[np.arange(len(query_indices)), query_indices] = -np.inf
        neighbors = np.argpartition(sim, -k, axis=1)[:, -k:]
        tgt_neighbors = tgt[neighbors]
        centroids = np.mean(tgt_neighbors, axis=1)
        dispersion = np.clip(1.0 - np.sum(centroids ** 2, axis=1), 0.0, None)
        entropies = 0.5 * (1.0 + np.log(2.0 * np.pi * np.e * (dispersion + epsilon)))
        avg_entropy = float(np.mean(entropies))
        branching_factor = float(np.exp(avg_entropy))
        return entropies, avg_entropy, branching_factor

    # 1. Observed metrics
    _, avg_ent_f, branch_f = compute_directed_metrics(X, Y)
    _, avg_ent_r, branch_r = compute_directed_metrics(Y, X)
    diff_obs = avg_ent_f - avg_ent_r

    # 2. Permutation test using fast precomputed dot products
    perm_diffs = []
    for _ in range(B):
        swap_mask = np.random.rand(N) < 0.5
        inv_swap_mask = ~swap_mask
        
        q_mask = swap_mask[query_indices]
        
        # Forward: X_perm as source, Y_perm as target
        sim_f = np.where(q_mask[:, None], np.where(swap_mask, dot_YY, dot_YX), np.where(swap_mask, dot_XY, dot_XX))
        sim_f[np.arange(len(query_indices)), query_indices] = -np.inf
        neighbors_f = np.argpartition(sim_f, -k, axis=1)[:, -k:]
        tgt_neighbors_f = xy_stacked[inv_swap_mask[neighbors_f].astype(int), neighbors_f]
        centroids_f = np.mean(tgt_neighbors_f, axis=1)
        dispersion_f = np.clip(1.0 - np.sum(centroids_f ** 2, axis=1), 0.0, None)
        avg_ent_f_perm = 0.5 * (1.0 + np.mean(np.log(2.0 * np.pi * np.e * (dispersion_f + epsilon))))
        
        # Reverse: Y_perm as source, X_perm as target
        inv_q_mask = ~q_mask
        sim_r = np.where(inv_q_mask[:, None], np.where(inv_swap_mask, dot_YY, dot_YX), np.where(inv_swap_mask, dot_XY, dot_XX))
        sim_r[np.arange(len(query_indices)), query_indices] = -np.inf
        neighbors_r = np.argpartition(sim_r, -k, axis=1)[:, -k:]
        tgt_neighbors_r = xy_stacked[swap_mask[neighbors_r].astype(int), neighbors_r]
        centroids_r = np.mean(tgt_neighbors_r, axis=1)
        dispersion_r = np.clip(1.0 - np.sum(centroids_r ** 2, axis=1), 0.0, None)
        avg_ent_r_perm = 0.5 * (1.0 + np.mean(np.log(2.0 * np.pi * np.e * (dispersion_r + epsilon))))
        
        perm_diffs.append(avg_ent_f_perm - avg_ent_r_perm)
        
    perm_diffs = np.array(perm_diffs)
    
    # Two-sided empirical p-value
    p_val = float((1.0 + np.sum(np.abs(perm_diffs) >= np.abs(diff_obs))) / (1.0 + B))
    
    return {
        "entropy_forward": avg_ent_f,
        "entropy_reverse": avg_ent_r,
        "branching_forward": branch_f,
        "branching_reverse": branch_r,
        "diff": diff_obs,
        "pvalue": p_val,
    }


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
    # H1 Test: Structural Transition (Multivariate Energy Distance)
    #
    # H1 tests whether the 6D distribution of transition features
    # (3 sequential operator norms + 3 pairwise cosines) differs between
    # the observed aspect pairing and a within-paper shuffled null.
    # A significant energy distance indicates structured epistemic trajectories.
    #
    # Primary test statistic: energy distance D² (multivariate, distribution-free).
    #   Observed D² is computed between F_obs (true pairing) and F_shuf (one
    #   shuffled draw).  Under H0 (exchangeability) we can relabel which rows
    #   belong to which group and recompute D² — the pooled permutation test.
    #
    # Secondary: per-edge Wasserstein effect sizes and KS diagnostics.
    # Complementary: 6-edge tetrahedron norms stored as features (h1_edge_norms).
    # -----------------------------------------------------------------------
    N = emb_p.shape[0]
    H1_SUBSAMPLE_MAX = 1000
    rng_perm = np.random.default_rng(42)

    # Observed operators (sequential + cross for 6-edge profile)
    pm = emb_m - emb_p
    mf = emb_f - emb_m
    fi = emb_i - emb_f
    pf = emb_f - emb_p
    pi = emb_i - emb_p
    mi = emb_i - emb_m

    # Sequential norms + cosines — 6D feature space for energy test
    norm_pm = np.linalg.norm(pm, axis=1)
    norm_mf = np.linalg.norm(mf, axis=1)
    norm_fi = np.linalg.norm(fi, axis=1)

    def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(sk_normalize(a) * sk_normalize(b), axis=1)

    cos_pm_mf = row_cos(pm, mf)
    cos_pm_fi = row_cos(pm, fi)
    cos_mf_fi = row_cos(mf, fi)

    F_obs = np.column_stack([
        norm_pm, norm_mf, norm_fi,
        cos_pm_mf, cos_pm_fi, cos_mf_fi,
    ])

    # Shuffled null (within-paper, one draw)
    shuffled_p = emb_p[rng_perm.permutation(N)]
    shuffled_m = emb_m[rng_perm.permutation(N)]
    shuffled_f = emb_f[rng_perm.permutation(N)]
    shuffled_i = emb_i[rng_perm.permutation(N)]

    pm_s = shuffled_m - shuffled_p
    mf_s = shuffled_f - shuffled_m
    fi_s = shuffled_i - shuffled_f

    norm_pm_s = np.linalg.norm(pm_s, axis=1)
    norm_mf_s = np.linalg.norm(mf_s, axis=1)
    norm_fi_s = np.linalg.norm(fi_s, axis=1)
    cos_pm_mf_s = row_cos(pm_s, mf_s)
    cos_pm_fi_s = row_cos(pm_s, fi_s)
    cos_mf_fi_s = row_cos(mf_s, fi_s)

    F_shuf = np.column_stack([
        norm_pm_s, norm_mf_s, norm_fi_s,
        cos_pm_mf_s, cos_pm_fi_s, cos_mf_fi_s,
    ])

    # Subsample for large N (energy test on 6D has strong power even with 2K points)
    if N > H1_SUBSAMPLE_MAX:
        sub_idx = rng_perm.choice(N, size=H1_SUBSAMPLE_MAX, replace=False)
        F_obs = F_obs[sub_idx]
        F_shuf = F_shuf[sub_idx]
        N_sub = H1_SUBSAMPLE_MAX
    else:
        N_sub = N

    # Energy distance between observed and shuffled distributions
    Z = np.vstack([F_obs, F_shuf])
    labels = np.array([0] * N_sub + [1] * N_sub)
    e_obs = energy_distance(Z[labels == 0], Z[labels == 1])
    metrics["h1_energy_stat"] = e_obs

    # Pooled permutation test (exchangeability under H0)
    B = 999
    count = 0
    for _ in range(B):
        rng_perm.shuffle(labels)
        e_perm = energy_distance(Z[labels == 0], Z[labels == 1])
        if e_perm >= e_obs:
            count += 1

    p_val = (count + 1) / (B + 1)
    metrics["h1_energy_pvalue"] = p_val

    # Per-edge Wasserstein effect sizes (1D, interpretable)
    metrics["h1_w_norm_pm"] = float(scipy_wasserstein_distance(norm_pm, norm_pm_s))
    metrics["h1_w_norm_mf"] = float(scipy_wasserstein_distance(norm_mf, norm_mf_s))
    metrics["h1_w_norm_fi"] = float(scipy_wasserstein_distance(norm_fi, norm_fi_s))
    metrics["h1_w_cos_pm_mf"] = float(scipy_wasserstein_distance(cos_pm_mf, cos_pm_mf_s))
    metrics["h1_w_cos_pm_fi"] = float(scipy_wasserstein_distance(cos_pm_fi, cos_pm_fi_s))
    metrics["h1_w_cos_mf_fi"] = float(scipy_wasserstein_distance(cos_mf_fi, cos_mf_fi_s))

    # KS diagnostics (secondary)
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

    # 6-edge tetrahedron norms for profile (complementary)
    norm_pf = np.linalg.norm(pf, axis=1)
    norm_pi = np.linalg.norm(pi, axis=1)
    norm_mi = np.linalg.norm(mi, axis=1)

    edge_norms = np.column_stack([
        norm_pm, norm_mf, norm_fi, norm_pf, norm_pi, norm_mi,
    ]).astype(np.float32)

    # Store features for re-analysis and dashboard
    features["h1_obs_features"] = F_obs.astype(np.float32)
    features["h1_shuf_features"] = F_shuf.astype(np.float32)
    features["h1_edge_norms"] = edge_norms

    # -----------------------------------------------------------------------
    # H2a Test: Local Transition Organization
    # -----------------------------------------------------------------------
    # Test for all 12 directional aspect-to-aspect transitions
    transitions = [
        ("pm", emb_p, emb_m),
        ("pf", emb_p, emb_f),
        ("pi", emb_p, emb_i),
        ("mp", emb_m, emb_p),
        ("mf", emb_m, emb_f),
        ("mi", emb_m, emb_i),
        ("fp", emb_f, emb_p),
        ("fm", emb_f, emb_m),
        ("fi", emb_f, emb_i),
        ("ip", emb_i, emb_p),
        ("im", emb_i, emb_m),
        ("if", emb_i, emb_f),
    ]
    for key, X_emb, Y_emb in transitions:
        w_obs, p_val, z_score = compute_h2_for_transition(X_emb, Y_emb)
        metrics[f"h2_w_dist_{key}"] = w_obs
        metrics[f"h2_pvalue_{key}"] = p_val
        metrics[f"h2_z_{key}"] = z_score

    # -----------------------------------------------------------------------
    # H2b Test: Local Transition Asymmetry
    # -----------------------------------------------------------------------
    asym_pairs = [
        ("pm", emb_p, emb_m),
        ("mf", emb_m, emb_f),
        ("fi", emb_f, emb_i),
        ("pf", emb_p, emb_f),
        ("pi", emb_p, emb_i),
        ("mi", emb_m, emb_i),
    ]
    for key, X_emb, Y_emb in asym_pairs:
        res_h2b = compute_h2b_for_pair(X_emb, Y_emb)
        metrics[f"h2b_entropy_forward_{key}"] = res_h2b["entropy_forward"]
        metrics[f"h2b_entropy_reverse_{key}"] = res_h2b["entropy_reverse"]
        metrics[f"h2b_branching_forward_{key}"] = res_h2b["branching_forward"]
        metrics[f"h2b_branching_reverse_{key}"] = res_h2b["branching_reverse"]
        metrics[f"h2b_diff_{key}"] = res_h2b["diff"]
        metrics[f"h2b_pvalue_{key}"] = res_h2b["pvalue"]

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

    # Fixed subsample for all H3 Wasserstein calls (eliminates subsampling variance
    # from the permutation p-value — same positions used for observed and null)
    H3_SUBSAMPLE = 2000
    rng_sub = np.random.RandomState(42)
    n_hist = P_hist.shape[0]
    n_pred = P_pred.shape[0]
    sub_hist = rng_sub.choice(n_hist, size=min(H3_SUBSAMPLE, n_hist), replace=False)
    sub_fut = rng_sub.choice(n_pred, size=min(H3_SUBSAMPLE, n_pred), replace=False)

    # Global Wasserstein evaluation (exact EMD with fixed subsample)
    w_edel = compute_wasserstein(P_pred, P_fut, idx_X=sub_fut, idx_Y=sub_fut)
    w_baseline = compute_wasserstein(P_hist, P_fut, idx_X=sub_hist, idx_Y=sub_fut)
    obs_gain = w_baseline - w_edel

    metrics["h3_w_edel"] = w_edel
    metrics["h3_w_baseline"] = w_baseline
    metrics["h3_predictive_gain"] = float(obs_gain)

    # Temporal permutation significance test for H3
    B_h3 = 49
    rng = np.random.default_rng(42)
    shuf_gains = []
    
    for _ in range(B_h3):
        shuf_idx = rng.permutation(N)
        hist_idx_b = shuf_idx[:n_hist]
        fut_idx_b = shuf_idx[n_hist:]
        
        I_hist_b = emb_i[hist_idx_b]
        P_hist_b = emb_p[hist_idx_b]
        I_fut_b = emb_i[fut_idx_b]
        P_fut_b = emb_p[fut_idx_b]
        
        reg_b = Ridge(alpha=1.0)
        reg_b.fit(I_hist_b, P_hist_b)
        P_pred_b = reg_b.predict(I_fut_b)
        
        w_edel_b = compute_wasserstein(P_pred_b, P_fut_b, idx_X=sub_fut, idx_Y=sub_fut)
        w_baseline_b = compute_wasserstein(P_hist_b, P_fut_b, idx_X=sub_hist, idx_Y=sub_fut)
        shuf_gains.append(w_baseline_b - w_edel_b)
        
    shuf_gains = np.array(shuf_gains)
    h3_gain_pvalue = float((1 + np.sum(shuf_gains >= obs_gain)) / (B_h3 + 1))
    metrics["h3_gain_pvalue"] = h3_gain_pvalue

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

    # Centroid 2D projection
    centroids_2d = None
    proj_x_cols = [c for c in df.columns if c.startswith("proj_") and c.endswith("_x")]
    if proj_x_cols:
        col_x = proj_x_cols[0]
        col_y = col_x[:-2] + "_y"
        if col_y in df.columns:
            labels = kmeans.labels_
            centroids_2d = np.zeros((n_clusters, 2))
            for c_idx in range(n_clusters):
                mask = (labels == c_idx)
                if mask.any():
                    centroids_2d[c_idx, 0] = df.loc[mask, col_x].mean()
                    centroids_2d[c_idx, 1] = df.loc[mask, col_y].mean()
                else:
                    centroids_2d[c_idx] = centroids[c_idx][:2]
    if centroids_2d is None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        centroids_2d = pca.fit_transform(centroids)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_x = x - x_mean
    z_y = y - y_mean
    x_std = np.std(x)
    y_std = np.std(y)
    z_x_std = z_x / x_std if x_std > 0 else z_x
    z_y_std = z_y / y_std if y_std > 0 else z_y
    lag_z_y = w @ z_y
    lag_z_y_std = w @ z_y_std

    features["h3_moran"] = {
        "x_raw": x.astype(np.float32),
        "y_raw": y.astype(np.float32),
        "z_x": z_x.astype(np.float32),
        "z_y": z_y.astype(np.float32),
        "lag_z_y": lag_z_y.astype(np.float32),
        "z_x_std": z_x_std.astype(np.float32),
        "lag_z_y_std": lag_z_y_std.astype(np.float32),
        "centroids_2d": centroids_2d.astype(np.float32),
        "moran_i": float(I_obs),
    }

    # Moran's I permutation significance
    B_moran = 19
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

    return {"metrics": metrics, "features": features}
