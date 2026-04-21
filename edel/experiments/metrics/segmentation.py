"""Metrics for epistemic segmentation quality.

Computes text-level statistics from the structured abstracts DataFrame.
Input: artifacts["structuring"] — DataFrame with columns:
    problem, method, finding, interpretation, abstract_text

Returns:
    {
        "metrics": {scalar stats},
        "features": {"seg_ratio_dist": np.ndarray(N,)}
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_ASPECTS = ["problem", "method", "finding", "interpretation"]
_ASPECT_PAIRS = [
    ("problem", "method"),
    ("problem", "finding"),
    ("problem", "interpretation"),
    ("method", "finding"),
    ("method", "interpretation"),
    ("finding", "interpretation"),
]


def segmentation_metrics(artifacts: dict) -> dict:
    """Compute segmentation quality metrics from structured abstracts."""
    df: pd.DataFrame = artifacts.get("structuring")
    if df is None or df.empty:
        return {"metrics": {}, "features": {}}

    # Only keep rows that have all 4 aspects and abstract_text
    required = _ASPECTS + ["abstract_text"]
    df = df.dropna(subset=required).copy()
    if df.empty:
        return {"metrics": {}, "features": {}}

    metrics: dict = {}

    # ── Aspect duplication rates ─────────────────────────────────────────────
    # Fraction of papers where two aspects are identical (copy-paste or LLM failure)
    for col_a, col_b in _ASPECT_PAIRS:
        key = f"dup_{col_a}_{col_b}"
        metrics[key] = float((df[col_a] == df[col_b]).mean())

    # ── Mean word length per aspect ──────────────────────────────────────────
    for col in _ASPECTS:
        lengths = df[col].fillna("").str.split().str.len()
        metrics[f"len_{col}"] = float(lengths.mean())

    # ── Segmentation ratio: segmented total / abstract length ────────────────
    seg_total = sum(
        df[col].fillna("").str.split().str.len() for col in _ASPECTS
    )
    abstract_len = df["abstract_text"].fillna("").str.split().str.len()

    # Avoid divide-by-zero
    ratio = np.where(abstract_len > 0, seg_total / abstract_len, np.nan)
    valid = ratio[~np.isnan(ratio)]

    metrics["seg_ratio_mean"] = float(np.mean(valid)) if len(valid) else float("nan")
    metrics["seg_ratio_std"] = float(np.std(valid)) if len(valid) else float("nan")
    metrics["abstract_len_mean"] = float(abstract_len.mean())
    metrics["seg_total_mean"] = float(seg_total.mean())
    metrics["dataset_size"] = int(len(df))

    # ── Features (distributions) ─────────────────────────────────────────────
    features = {
        "seg_ratio_dist": ratio.astype(np.float32),
    }

    return {"metrics": metrics, "features": features}
