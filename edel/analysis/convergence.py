"""Convergence analysis and calibration module for EDEL."""

from __future__ import annotations

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize as sk_normalize

from edel.io.artifact import make_stage_artifact
from edel.experiments.registry import get_experiment
from edel.pipeline.projection import load_embeddings_to_matrix
from edel.experiments.metrics.hypothesis_tests import compute_wasserstein, compute_h2_for_transition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling Helpers
# ---------------------------------------------------------------------------

def sample_temporal_stratified(df: pd.DataFrame, n: int, random_state: int | None = None) -> list[int]:
    """Sample n indices from df preserving the proportions of publication_year."""
    N = len(df)
    if n >= N:
        return list(range(N))

    rng = np.random.default_rng(random_state)
    
    # Extract publication years
    years = df["publication_year"].copy()
    valid_years = []
    for y in years.values:
        try:
            valid_years.append(int(float(y)))
        except (ValueError, TypeError):
            valid_years.append(-1)
    valid_years = np.array(valid_years)
    
    unique_years, counts = np.unique(valid_years[valid_years > 0], return_counts=True)
    
    if len(unique_years) == 0:
        # Fallback to uniform if no valid years
        return rng.choice(N, size=n, replace=False).tolist()

    # Calculate proportions
    total_valid = counts.sum()
    proportions = counts / total_valid
    
    # Calculate target counts per year
    target_counts = np.round(n * proportions).astype(int)
    
    # Handle rounding issues to make sure sum(target_counts) == n
    diff = n - target_counts.sum()
    if diff != 0:
        # Adjust the largest category or randomly distributed
        adj_idx = np.argsort(target_counts)[-1]
        target_counts[adj_idx] = max(0, target_counts[adj_idx] + diff)
        
    sampled_indices = []
    
    # Track indices for valid years
    for yr, t_count in zip(unique_years, target_counts):
        if t_count <= 0:
            continue
        yr_indices = np.where(valid_years == yr)[0]
        if len(yr_indices) <= t_count:
            # If not enough elements, take all of them and we will pad later if needed
            sampled_indices.extend(yr_indices.tolist())
        else:
            sampled_indices.extend(rng.choice(yr_indices, size=t_count, replace=False).tolist())
            
    # If we are slightly short of n due to small year bins, fill randomly from the rest
    remaining = n - len(sampled_indices)
    if remaining > 0:
        all_indices = set(range(N))
        already_sampled = set(sampled_indices)
        candidate_indices = list(all_indices - already_sampled)
        if candidate_indices:
            fill_size = min(remaining, len(candidate_indices))
            sampled_indices.extend(rng.choice(candidate_indices, size=fill_size, replace=False).tolist())
            
    return sampled_indices[:n]


# ---------------------------------------------------------------------------
# Core Analytical Functions
# ---------------------------------------------------------------------------

def run_convergence_analysis(
    experiment_id: str, 
    base_path: str | Path = "artifacts",
    force: bool = False
) -> dict:
    """Execute H1, H2, and H3 convergence study for a given experiment config."""
    base_path = Path(base_path)
    
    # Check cache first
    cache_path = base_path / "experiments" / experiment_id / "convergence_results.pkl"
    if cache_path.exists() and not force:
        logger.info(f"Loading cached convergence results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
            
    logger.info(f"Starting convergence study for {experiment_id}...")
    
    # Load experiment configuration
    config = get_experiment(experiment_id)
    
    # Load dataframe (try embeddings, fallback to clustering)
    df = None
    art_emb = make_stage_artifact(config, base_path, "embeddings", "embeddings")
    if art_emb.parquet_path.exists():
        df = pd.read_parquet(art_emb.parquet_path)
    else:
        art_clust = make_stage_artifact(config, base_path, "clustering", "clustering")
        if art_clust.parquet_path.exists():
            df = pd.read_parquet(art_clust.parquet_path)
            
    if df is None:
        raise FileNotFoundError(f"No processed embeddings or clustering file found for config {experiment_id}")
        
    N = len(df)
    logger.info(f"Loaded dataset with N={N} records.")
    
    dimensions = config.get("embedding", {}).get("n_dimensions", 1536)
    aspects = ["problem", "method", "finding", "interpretation"]
    if not all(f"{a}_embedding" in df.columns for a in aspects):
        raise ValueError("Missing aspect embeddings in the dataset.")
        
    # Extract and normalize matrices
    def load_mat(aspect: str) -> np.ndarray:
        mat = load_embeddings_to_matrix(df, f"{aspect}_embedding", dimensions)
        mat -= mat.mean(axis=0)
        return sk_normalize(mat)
        
    emb_p = load_mat("problem")
    emb_m = load_mat("method")
    emb_f = load_mat("finding")
    emb_i = load_mat("interpretation")
    
    # Precompute transition elements for full dataset
    pm_full = emb_m - emb_p
    mf_full = emb_f - emb_m
    fi_full = emb_i - emb_f
    
    norm_pm_full = np.linalg.norm(pm_full, axis=1)
    norm_mf_full = np.linalg.norm(mf_full, axis=1)
    norm_fi_full = np.linalg.norm(fi_full, axis=1)
    
    def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(sk_normalize(a) * sk_normalize(b), axis=1)
        
    cos_pm_mf_full = row_cos(pm_full, mf_full)
    cos_pm_fi_full = row_cos(pm_full, fi_full)
    cos_mf_fi_full = row_cos(mf_full, fi_full)
    
    # Run full H1 to get reference KS values
    shuf_idx = np.random.permutation(N)
    pm_full_s = emb_m[shuf_idx] - emb_p[shuf_idx]
    mf_full_s = emb_f[shuf_idx] - emb_m[shuf_idx]
    fi_full_s = emb_i[shuf_idx] - emb_f[shuf_idx]
    
    norm_pm_full_s = np.linalg.norm(pm_full_s, axis=1)
    norm_mf_full_s = np.linalg.norm(mf_full_s, axis=1)
    norm_fi_full_s = np.linalg.norm(fi_full_s, axis=1)
    
    cos_pm_mf_full_s = row_cos(pm_full_s, mf_full_s)
    cos_pm_fi_full_s = row_cos(pm_full_s, fi_full_s)
    cos_mf_fi_full_s = row_cos(mf_full_s, fi_full_s)
    
    h1_full_refs = {
        "norm_pm": float(ks_2samp(norm_pm_full, norm_pm_full_s).statistic),
        "norm_mf": float(ks_2samp(norm_mf_full, norm_mf_full_s).statistic),
        "norm_fi": float(ks_2samp(norm_fi_full, norm_fi_full_s).statistic),
        "cos_pm_mf": float(ks_2samp(cos_pm_mf_full, cos_pm_mf_full_s).statistic),
        "cos_pm_fi": float(ks_2samp(cos_pm_fi_full, cos_pm_fi_full_s).statistic),
        "cos_mf_fi": float(ks_2samp(cos_mf_fi_full, cos_mf_fi_full_s).statistic),
    }
    
    # Run full H2 to get reference z-scores (using fast setting to keep analysis fast)
    h2_transitions = [
        ("pm", emb_p, emb_m), ("pf", emb_p, emb_f), ("pi", emb_p, emb_i),
        ("mp", emb_m, emb_p), ("mf", emb_m, emb_f), ("mi", emb_m, emb_i),
        ("fp", emb_f, emb_p), ("fm", emb_f, emb_m), ("fi", emb_f, emb_i),
        ("ip", emb_i, emb_p), ("im", emb_i, emb_m), ("if", emb_i, emb_f)
    ]
    
    h2_full_z = {}
    h2_full_sig = []
    for key, X_emb, Y_emb in h2_transitions:
        _, p_val, z_score = compute_h2_for_transition(X_emb, Y_emb, B=100, max_queries=25)
        h2_full_z[key] = z_score
        if p_val < 0.05:
            h2_full_sig.append(key)
            
    # H3 time split and evaluation on full dataset
    years = df["publication_year"].copy()
    valid_years = []
    for y in years.values:
        try:
            valid_years.append(int(float(y)))
        except (ValueError, TypeError):
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
        
    I_hist = emb_i[hist_mask]
    P_hist = emb_p[hist_mask]
    I_fut = emb_i[fut_mask]
    P_fut = emb_p[fut_mask]
    
    reg_full = Ridge(alpha=1.0)
    reg_full.fit(I_hist, P_hist)
    P_pred_full = reg_full.predict(I_fut)
    
    # Cap size for speed
    def get_w_dist(X, Y):
        if len(X) > 1000:
            idx_x = np.random.choice(len(X), size=1000, replace=False)
            X = X[idx_x]
        if len(Y) > 1000:
            idx_y = np.random.choice(len(Y), size=1000, replace=False)
            Y = Y[idx_y]
        return compute_wasserstein(X, Y)
        
    w_edel_full = get_w_dist(P_pred_full, P_fut)
    w_base_full = get_w_dist(P_hist, P_fut)
    h3_full_gain = w_base_full - w_edel_full
    
    # Define sample parameters
    R = 20
    if N < 5000:
        h1_percentages = [0.05, 0.10, 0.20, 0.30, 0.50]
        h1_sizes = [int(np.round(p * N)) for p in h1_percentages]
        h1_sizes = sorted(list(set([sz for sz in h1_sizes if 10 < sz < N])))
        if not h1_sizes:
            h1_sizes = [min(50, N // 2)]
    else:
        h1_sizes = [500, 1000, 2000, 3000, 5000, 10000]
        h1_sizes = [size for size in h1_sizes if size < N]
    
    # -----------------------------------------------------------------------
    # H1 & H2 Convergence Execution
    # -----------------------------------------------------------------------
    h1_results = {size: {"ks_stat": {k: [] for k in h1_full_refs.keys()}, 
                         "ks_pval": {k: [] for k in h1_full_refs.keys()},
                         "w_dist": {k: [] for k in h1_full_refs.keys()}} for size in h1_sizes}
                         
    h2_results = {size: {"mae_z": [], "jaccard": []} for size in h1_sizes}
    
    rng = np.random.default_rng(42)
    
    for size in h1_sizes:
        logger.info(f"Running H1 & H2 Convergence for size={size}...")
        for rep in range(R):
            # Sample indices
            sample_idx = rng.choice(N, size=size, replace=False)
            
            # --- H1 Sub-calculations ---
            emb_p_s = emb_p[sample_idx]
            emb_m_s = emb_m[sample_idx]
            emb_f_s = emb_f[sample_idx]
            emb_i_s = emb_i[sample_idx]
            
            pm_s = emb_m_s - emb_p_s
            mf_s = emb_f_s - emb_m_s
            fi_s = emb_i_s - emb_f_s
            
            norm_pm_s = np.linalg.norm(pm_s, axis=1)
            norm_mf_s = np.linalg.norm(mf_s, axis=1)
            norm_fi_s = np.linalg.norm(fi_s, axis=1)
            
            cos_pm_mf_s = row_cos(pm_s, mf_s)
            cos_pm_fi_s = row_cos(pm_s, fi_s)
            cos_mf_fi_s = row_cos(mf_s, fi_s)
            
            # Shuffle targets within the sample
            shuf_sample_idx = rng.permutation(size)
            pm_shuf = emb_m_s[shuf_sample_idx] - emb_p_s[shuf_sample_idx]
            mf_shuf = emb_f_s[shuf_sample_idx] - emb_m_s[shuf_sample_idx]
            fi_shuf = emb_i_s[shuf_sample_idx] - emb_f_s[shuf_sample_idx]
            
            norm_pm_shuf = np.linalg.norm(pm_shuf, axis=1)
            norm_mf_shuf = np.linalg.norm(mf_shuf, axis=1)
            norm_fi_shuf = np.linalg.norm(fi_shuf, axis=1)
            
            cos_pm_mf_shuf = row_cos(pm_shuf, mf_shuf)
            cos_pm_fi_shuf = row_cos(pm_shuf, fi_shuf)
            cos_mf_fi_shuf = row_cos(mf_shuf, fi_shuf)
            
            # Helper to run KS and 1D Wasserstein
            def calc_metrics(obs, null, full_obs, prefix):
                ks_res = ks_2samp(obs, null)
                h1_results[size]["ks_stat"][prefix].append(float(ks_res.statistic))
                h1_results[size]["ks_pval"][prefix].append(float(ks_res.pvalue))
                
                # 1D Wasserstein distance between sample observed distribution and full observed distribution
                w_dist = float(scipy_wasserstein_distance(obs, full_obs))
                h1_results[size]["w_dist"][prefix].append(w_dist)
                
            calc_metrics(norm_pm_s, norm_pm_shuf, norm_pm_full, "norm_pm")
            calc_metrics(norm_mf_s, norm_mf_shuf, norm_mf_full, "norm_mf")
            calc_metrics(norm_fi_s, norm_fi_shuf, norm_fi_full, "norm_fi")
            calc_metrics(cos_pm_mf_s, cos_pm_mf_shuf, cos_pm_mf_full, "cos_pm_mf")
            calc_metrics(cos_pm_fi_s, cos_pm_fi_shuf, cos_pm_fi_full, "cos_pm_fi")
            calc_metrics(cos_mf_fi_s, cos_mf_fi_shuf, cos_mf_fi_full, "cos_mf_fi")
            
            # --- H2 Sub-calculations ---
            h2_sample_z = {}
            h2_sample_sig = []
            
            for key, X_full, Y_full in h2_transitions:
                X_s = X_full[sample_idx]
                Y_s = Y_full[sample_idx]
                _, p_val, z_score = compute_h2_for_transition(X_s, Y_s, B=100, max_queries=25)
                h2_sample_z[key] = z_score
                if p_val < 0.05:
                    h2_sample_sig.append(key)
                    
            # Compute MAE z
            mae_z = np.mean([abs(h2_sample_z[k] - h2_full_z[k]) for k in h2_sample_z])
            h2_results[size]["mae_z"].append(float(mae_z))
            
            # Compute Jaccard
            union = set(h2_full_sig) | set(h2_sample_sig)
            if not union:
                jaccard = 1.0
            else:
                inter = set(h2_full_sig) & set(h2_sample_sig)
                jaccard = len(inter) / len(union)
            h2_results[size]["jaccard"].append(jaccard)
            
    # -----------------------------------------------------------------------
    # H3 Calibration Execution
    # -----------------------------------------------------------------------
    h3_percentages = [0.05, 0.10, 0.20, 0.30, 0.50]
    h3_sizes = [int(np.round(p * N)) for p in h3_percentages]
    h3_valid = [(p, sz) for p, sz in zip(h3_percentages, h3_sizes) if 5 < sz < N]
    if not h3_valid:
        h3_valid = [(0.5, min(50, N // 2))]
    h3_percentages = [item[0] for item in h3_valid]
    h3_sizes = [item[1] for item in h3_valid]
    
    h3_results = {
        "percentages": h3_percentages,
        "sizes": h3_sizes,
        "Scheme A": {size: [] for size in h3_sizes},
        "Scheme B": {size: [] for size in h3_sizes}
    }
    
    for pct, size in zip(h3_percentages, h3_sizes):
        logger.info(f"Running H3 Calibration for size={size} ({pct*100}%)...")
        for rep in range(R):
            # --- Scheme A (Uniform Random) ---
            idx_a = rng.choice(N, size=size, replace=False)
            
            # --- Scheme B (Temporal Stratified) ---
            idx_b = sample_temporal_stratified(df, size, random_state=42 + rep)
            
            def evaluate_h3_sample(sample_indices):
                years_s = valid_years[sample_indices]
                hist_mask_s = (years_s > 0) & (years_s <= split_year)
                fut_mask_s = (years_s > 0) & (years_s > split_year)
                
                # Handle tiny splits in sample
                if hist_mask_s.sum() < 5 or fut_mask_s.sum() < 5:
                    split_idx_s = len(sample_indices) // 2
                    hist_mask_s = np.zeros(len(sample_indices), dtype=bool)
                    hist_mask_s[:split_idx_s] = True
                    fut_mask_s = ~hist_mask_s
                    
                I_h = emb_i[sample_indices][hist_mask_s]
                P_h = emb_p[sample_indices][hist_mask_s]
                I_f = emb_i[sample_indices][fut_mask_s]
                P_f = emb_p[sample_indices][fut_mask_s]
                
                reg = Ridge(alpha=1.0)
                reg.fit(I_h, P_h)
                P_pred = reg.predict(I_f)
                
                w_edel = get_w_dist(P_pred, P_f)
                w_base = get_w_dist(P_h, P_f)
                gain = w_base - w_edel
                return float(gain)
                
            h3_results["Scheme A"][size].append(evaluate_h3_sample(idx_a))
            h3_results["Scheme B"][size].append(evaluate_h3_sample(idx_b))
            
    # Construct combined package
    results = {
        "experiment_id": experiment_id,
        "N": N,
        "h1_results": {
            "sample_sizes": h1_sizes,
            "data": h1_results,
            "full_refs": h1_full_refs
        },
        "h2_results": {
            "sample_sizes": h1_sizes,
            "data": h2_results,
            "full_z": h2_full_z,
            "full_sig": h2_full_sig
        },
        "h3_results": {
            "percentages": h3_percentages,
            "sizes": h3_sizes,
            "data": {
                "Scheme A": h3_results["Scheme A"],
                "Scheme B": h3_results["Scheme B"]
            },
            "full_gain": h3_full_gain
        }
    }
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
        
    logger.info(f"Convergence study complete and cached at {cache_path}!")
    return results
