"""Operator metrics: epistemic transition norms, angles, and 6-dim feature matrix.

Operators are the difference vectors between consecutive aspect embeddings:
    pm = method  - problem
    mf = finding - method
    fi = interp  - finding

The 6-dim feature matrix is [cos(pm,mf), cos(pm,fi), cos(mf,fi), |pm|, |mf|, |fi|]
per paper — used downstream in structure_metrics and KS comparison tests.

Input: artifacts["embedding"] — DataFrame with aspect embedding columns.

Side-effect: writes computed operators into the shared context dict so that
structure_metrics (which depends on pm/mf/fi) can reuse them without
re-loading embeddings.

Returns:
    {
        "metrics": {scalar stats},
        "features": {per-paper distributions + "transition_features": (N,6)}
    }
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_normalize

from edel.pipeline.projection import load_embeddings_to_matrix


def operator_metrics(artifacts: dict) -> dict:
    """Compute epistemic operator metrics and 6-dim paper-style feature matrix."""
    df: pd.DataFrame = artifacts.get("embedding")
    if df is None:
        return {"metrics": {}, "features": {}}

    dimensions = artifacts.get("_dimensions")
    if dimensions is None:
        from edel.pipeline.projection import detect_embedding_dimensions
        dimensions = detect_embedding_dimensions(df, {})
    aspects = ["problem", "method", "finding", "interpretation"]
    if not all(f"{a}_embedding" in df.columns for a in aspects):
        return {"metrics": {}, "features": {}}

    # ── Load and centre-normalize embedding matrices ─────────────────────────
    def load(aspect: str) -> np.ndarray:
        mat = load_embeddings_to_matrix(df, f"{aspect}_embedding", dimensions)
        mat -= mat.mean(axis=0)             # centre
        return sk_normalize(mat)            # unit norm

    emb_p = load("problem")
    emb_m = load("method")
    emb_f = load("finding")
    emb_i = load("interpretation")

    # ── Operators (difference vectors) ───────────────────────────────────────
    pm = emb_m - emb_p
    mf = emb_f - emb_m
    fi = emb_i - emb_f
    pf = emb_f - emb_p
    pi = emb_i - emb_p
    mi = emb_i - emb_m

    # Normalised versions for cosine computation
    pm_n = sk_normalize(pm)
    mf_n = sk_normalize(mf)
    fi_n = sk_normalize(fi)

    def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Row-wise cosine similarity of already-normalised vectors."""
        return np.sum(a * b, axis=1)

    metrics: dict = {}
    features: dict = {}

    # ── Scalar metrics ───────────────────────────────────────────────────────
    norm_pm = np.linalg.norm(pm, axis=1)
    norm_mf = np.linalg.norm(mf, axis=1)
    norm_fi = np.linalg.norm(fi, axis=1)

    metrics["norm_pm"] = float(norm_pm.mean())
    metrics["norm_mf"] = float(norm_mf.mean())
    metrics["norm_fi"] = float(norm_fi.mean())
    metrics["norm_pf"] = float(np.linalg.norm(pf, axis=1).mean())
    metrics["norm_pi"] = float(np.linalg.norm(pi, axis=1).mean())
    metrics["norm_mi"] = float(np.linalg.norm(mi, axis=1).mean())

    cos_pm_mf = row_cos(pm_n, mf_n)
    cos_pm_fi = row_cos(pm_n, fi_n)
    cos_mf_fi = row_cos(mf_n, fi_n)

    metrics["cos_pm_mf"] = float(cos_pm_mf.mean())
    metrics["cos_pm_fi"] = float(cos_pm_fi.mean())
    metrics["cos_mf_fi"] = float(cos_mf_fi.mean())

    # ── Cycle Closure & Net Epistemic Displacement ───────────────────────────
    cycle_closure_dist = np.linalg.norm(emb_i - emb_p, axis=1)
    metrics["cycle_closure_norm"] = float(cycle_closure_dist.mean())

    N_papers = emb_i.shape[0]
    k_nn = min(5, N_papers - 1)
    if k_nn > 0:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine", n_jobs=min(os.cpu_count() or 1, 4))
        nn.fit(emb_i)
        _, indices = nn.kneighbors(emb_i)
        nn_indices = indices[:, 1:]
        bar_p = emb_p[nn_indices].mean(axis=1)
        net_displacement_dist = np.linalg.norm(bar_p - emb_p, axis=1)
    else:
        net_displacement_dist = np.zeros(N_papers)
    metrics["net_epistemic_displacement_norm"] = float(net_displacement_dist.mean())

    # ── Feature distributions (per-paper) ────────────────────────────────────
    features["norm_pm_dist"] = norm_pm.astype(np.float32)
    features["norm_mf_dist"] = norm_mf.astype(np.float32)
    features["norm_fi_dist"] = norm_fi.astype(np.float32)
    features["norm_pf_dist"] = np.linalg.norm(pf, axis=1).astype(np.float32)
    features["norm_pi_dist"] = np.linalg.norm(pi, axis=1).astype(np.float32)
    features["norm_mi_dist"] = np.linalg.norm(mi, axis=1).astype(np.float32)
    features["cos_pm_mf_dist"] = cos_pm_mf.astype(np.float32)
    features["cos_pm_fi_dist"] = cos_pm_fi.astype(np.float32)
    features["cos_mf_fi_dist"] = cos_mf_fi.astype(np.float32)
    features["cycle_closure_dist"] = cycle_closure_dist.astype(np.float32)
    features["net_displacement_dist"] = net_displacement_dist.astype(np.float32)

    # ── 6-dim transition feature matrix (N, 6) ───────────────────────────────
    transition_features = np.column_stack([
        cos_pm_mf, cos_pm_fi, cos_mf_fi,
        norm_pm,   norm_mf,   norm_fi,
    ]).astype(np.float32)

    features["transition_features"] = transition_features

    # ── Write operators into shared context for structure_metrics ─────────────
    # artifacts is the shared context dict — we write intermediates here
    artifacts["_operators"] = {"pm": pm, "mf": mf, "fi": fi}
    artifacts["_transition_features"] = transition_features

    return {"metrics": metrics, "features": features}
