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
            sim_mat = cosine_similarity(embs[col_a], embs[col_b])
            metrics[f"sep_{sa}_{sb}"] = float(sim_mat.mean())

    # ── Intra-aspect density: distribution of upper-triangle cosines ─────────
    for aspect in _ASPECTS:
        sa = _ASPECT_SHORT[aspect]
        sim_mat = cosine_similarity(embs[aspect])
        idx = np.triu_indices_from(sim_mat, k=1)
        upper = sim_mat[idx]

        # Sample if too many pairs
        if len(upper) > _MAX_DENSITY_PAIRS:
            rng = np.random.default_rng(0)
            upper = rng.choice(upper, size=_MAX_DENSITY_PAIRS, replace=False)

        metrics[f"density_{sa}_mean"] = float(upper.mean())
        metrics[f"density_{sa}_std"] = float(upper.std())
        features[f"density_{sa}_dist"] = upper.astype(np.float32)

    return {"metrics": metrics, "features": features}
