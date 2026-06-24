"""Embedding-level metrics: intra-aspect cosine similarity, separability, density.

Input: artifacts["embedding"] — DataFrame with aspect embedding columns.

Returns:
    {
        "metrics": {scalar stats},
        "features": {per-paper distributions}
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize

from edel.pipeline.projection import load_embeddings_to_matrix

_ASPECTS = ["problem", "method", "finding", "interpretation"]
_ASPECT_SHORT = {"problem": "p", "method": "m", "finding": "f", "interpretation": "i"}

_PAIRS = [
    ("problem", "method"),
    ("problem", "finding"),
    ("problem", "interpretation"),
    ("method", "finding"),
    ("method", "interpretation"),
    ("finding", "interpretation"),
]

# Maximum number of pairs sampled for O(N²) density metrics
_MAX_DENSITY_PAIRS = 50_000


def embedding_metrics(artifacts: dict, remove_pc: int = 0) -> dict:
    """Compute embedding-level metrics from aspect embedding columns."""
    df: pd.DataFrame = artifacts.get("embedding")
    if df is None:
        return {"metrics": {}, "features": {}}

    dimensions = artifacts.get("_dimensions", 1536)
    missing = [a for a in _ASPECTS if f"{a}_embedding" not in df.columns]
    if missing:
        return {"metrics": {}, "features": {}}

    # ── Load embedding matrices ──────────────────────────────────────────────
    embs = {
        a: sk_normalize(load_embeddings_to_matrix(df, f"{a}_embedding", dimensions))
        for a in _ASPECTS
    }

def apply_anisotropy_correction(embs_dict: dict[str, np.ndarray], method: str = "none", n_components: int = 0) -> dict[str, np.ndarray]:
    """Apply anisotropy correction to a dictionary of embeddings."""
    if method == "none" or (method == "pc_removal" and n_components <= 0):
        return embs_dict
        
    # Concatenate all available embeddings to find global stats (mean or PCs)
    all_embs = np.vstack([X for X in embs_dict.values()])
    
    corrected = {}
    if method == "pc_removal":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(all_embs)
        for a, X in embs_dict.items():
            pc_projection = pca.transform(X)
            X_corrected = X - pca.inverse_transform(pc_projection)
            corrected[a] = sk_normalize(X_corrected)
            
    elif method == "mean_centering":
        global_mean = all_embs.mean(axis=0)
        for a, X in embs_dict.items():
            X_corrected = X - global_mean
            corrected[a] = sk_normalize(X_corrected)
    else:
        return embs_dict
        
    return corrected


def compute_joint_space_metrics(
    matrices_dict: dict[str, np.ndarray],
    categories: list[str],
    k: int = 5
) -> dict:
    """Compute Silhouette score, 1-NN classification accuracy (excluding same paper),

    and nearest neighbor role ratios.
    """
    from sklearn.metrics import silhouette_score
    
    # Get N
    first_matrix = next(iter(matrices_dict.values()))
    N = first_matrix.shape[0]
    if N <= 1:
        return {}
        
    # Stack matrices: shape (len(categories) * N, D)
    # Make sure they are L2-normalized
    stacked = np.vstack([sk_normalize(matrices_dict[cat]) for cat in categories])
    
    # Pairwise cosine similarity and distance
    sim = np.dot(stacked, stacked.T)
    # Clip to avoid float precision issues
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    
    # 1. Silhouette score
    aspect_labels = np.array([cat for cat in categories for _ in range(N)])
    try:
        sil = float(silhouette_score(dist, aspect_labels, metric="precomputed"))
    except Exception:
        sil = 0.0
        
    # 2. 1-NN classification (excluding same paper) and NN ratios
    correct_1nn = 0
    total_points = len(stacked)
    
    same_paper_counts = 0.0
    same_cat_counts = 0.0
    other_counts = 0.0
    
    for i in range(total_points):
        # Sort indices by distance ascending
        sorted_indices = np.argsort(dist[i])
        # Exclude itself (which is at distance 0)
        sorted_indices = sorted_indices[sorted_indices != i]
        
        # A. 1-NN classification (first neighbor not from the same paper)
        pred_label = None
        for nbr in sorted_indices:
            if (nbr % N) != (i % N):
                pred_label = aspect_labels[nbr]
                break
        if pred_label == aspect_labels[i]:
            correct_1nn += 1
            
        # B. Nearest Neighbor ratios (top-k neighbors excluding itself)
        top_k_nbrs = sorted_indices[:k]
        actual_k = len(top_k_nbrs)
        if actual_k > 0:
            for nbr in top_k_nbrs:
                nbr_paper = nbr % N
                nbr_cat = nbr // N
                curr_paper = i % N
                curr_cat = i // N
                
                if nbr_paper == curr_paper:
                    same_paper_counts += 1.0 / actual_k
                elif nbr_cat == curr_cat:
                    same_cat_counts += 1.0 / actual_k
                else:
                    other_counts += 1.0 / actual_k
                    
    acc_1nn = float(correct_1nn / total_points)
    nn_same_paper = float(same_paper_counts / total_points)
    nn_same_cat = float(same_cat_counts / total_points)
    nn_other = float(other_counts / total_points)
    
    return {
        "silhouette": sil,
        "accuracy_1nn": acc_1nn,
        "nn_same_paper": nn_same_paper,
        "nn_same_category": nn_same_cat,
        "nn_other": nn_other
    }


def embedding_metrics(artifacts: dict, correction_method: str = "none", remove_pc: int = 0) -> dict:
    """Compute embedding-level metrics from aspect embedding columns."""
    df: pd.DataFrame = artifacts.get("embedding")
    if df is None:
        return {"metrics": {}, "features": {}}

    dimensions = artifacts.get("_dimensions", 1536)
    missing = [a for a in _ASPECTS if f"{a}_embedding" not in df.columns]
    if missing:
        return {"metrics": {}, "features": {}}

    # ── Load embedding matrices ──────────────────────────────────────────────
    embs = {
        a: sk_normalize(load_embeddings_to_matrix(df, f"{a}_embedding", dimensions))
        for a in _ASPECTS
    }

    # ── Optional: Anisotropy Correction ──────────────────────────────────────
    if correction_method != "none":
        embs = apply_anisotropy_correction(embs, method=correction_method, n_components=remove_pc)

    N = next(iter(embs.values())).shape[0]
    metrics: dict = {}
    features: dict = {}

    # ── Diagonal (per-paper) cosine similarities ─────────────────────────────
    # e.g. sim_pm = cosine(problem_i, method_i) averaged over all papers
    for col_a, col_b in _PAIRS:
        sa = _ASPECT_SHORT[col_a]
        sb = _ASPECT_SHORT[col_b]
        key = f"sim_{sa}{sb}"
        # Row-wise dot product of L2-normalised vectors = cosine similarity
        per_paper = np.sum(embs[col_a] * embs[col_b], axis=1)  # (N,)
        metrics[key] = float(per_paper.mean())
        features[f"{key}_dist"] = per_paper.astype(np.float32)

    # ── Cross-set separability (mean of full pairwise cosine matrix) ─────────
    # sep_p_m = mean(cosine(all problems, all methods)) — different papers included
    for col_a in _ASPECTS:
        for col_b in _ASPECTS:
            sa = _ASPECT_SHORT[col_a]
            sb = _ASPECT_SHORT[col_b]
            # Full N×N is fine for N≤500; for larger we'll add sampling later
            if N > 500:
                rng = np.random.default_rng(0)
                idx_a = rng.choice(N, size=500, replace=False)
                idx_b = rng.choice(N, size=500, replace=False)
                sim_mat = cosine_similarity(embs[col_a][idx_a], embs[col_b][idx_b])
            else:
                sim_mat = cosine_similarity(embs[col_a], embs[col_b])
            metrics[f"sep_{sa}_{sb}"] = float(sim_mat.mean())

    # ── Intra-aspect density: distribution of upper-triangle cosines ─────────
    for aspect in _ASPECTS:
        sa = _ASPECT_SHORT[aspect]
        if N * (N - 1) // 2 > _MAX_DENSITY_PAIRS:
            rng = np.random.default_rng(0)
            pairs = set()
            while len(pairs) < _MAX_DENSITY_PAIRS:
                needed = _MAX_DENSITY_PAIRS - len(pairs)
                i_candidates = rng.choice(N, size=int(needed * 1.1))
                j_candidates = rng.choice(N, size=int(needed * 1.1))
                for i, j in zip(i_candidates, j_candidates):
                    if i != j:
                        pairs.add((min(i, j), max(i, j)))
                        if len(pairs) == _MAX_DENSITY_PAIRS:
                            break
            pairs_list = list(pairs)
            i_idx = np.array([p[0] for p in pairs_list])
            j_idx = np.array([p[1] for p in pairs_list])
            upper = np.sum(embs[aspect][i_idx] * embs[aspect][j_idx], axis=1)
        else:
            sim_mat = cosine_similarity(embs[aspect])
            idx = np.triu_indices_from(sim_mat, k=1)
            upper = sim_mat[idx]

        metrics[f"density_{sa}_mean"] = float(upper.mean())
        metrics[f"density_{sa}_std"] = float(upper.std())
        features[f"density_{sa}_dist"] = upper.astype(np.float32)

    # ── Joint Space & Transition Space Overlap Metrics ───────────────────────
    joint_aspect_res = compute_joint_space_metrics(embs, _ASPECTS, k=5)
    for k_metric, v_metric in joint_aspect_res.items():
        metrics[f"joint_aspect_{k_metric}"] = v_metric

    trans_embs = {
        "PM": embs["method"] - embs["problem"],
        "MF": embs["finding"] - embs["method"],
        "FI": embs["interpretation"] - embs["finding"]
    }
    joint_trans_res = compute_joint_space_metrics(trans_embs, ["PM", "MF", "FI"], k=5)
    for k_metric, v_metric in joint_trans_res.items():
        metrics[f"joint_trans_{k_metric}"] = v_metric

    return {"metrics": metrics, "features": features}
