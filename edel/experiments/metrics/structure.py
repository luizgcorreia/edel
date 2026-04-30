"""Structure metrics: silhouette scores on operator clusters and feature clusters.

Depends on operator_metrics having run first — reads pm/mf/fi and the
transition_features matrix from the shared context dict (artifacts["_operators"]
and artifacts["_transition_features"]).

Returns:
    {
        "metrics": {"silhouette_transitions": float, "silhouette_features": float},
        "features": {}
    }
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def structure_metrics(artifacts: dict) -> dict:
    """Compute silhouette scores on epistemic operator and feature clusters."""
    operators = artifacts.get("_operators")
    transition_features = artifacts.get("_transition_features")

    if operators is None or transition_features is None:
        print(
            "Warning [structure_metrics]: operator intermediates not found in context. "
            "Ensure operator_metrics runs before structure_metrics in METRIC_REGISTRY."
        )
        return {"metrics": {}, "features": {}}

    pm = operators["pm"]
    mf = operators["mf"]
    fi = operators["fi"]

    metrics: dict = {}

    # ── Silhouette on stacked operators (3 classes: pm, mf, fi) ──────────────
    X_ops = np.vstack([pm, mf, fi])
    n = pm.shape[0]
    labels_ops = np.array([0] * n + [1] * n + [2] * n)

    try:
        kmeans_ops = KMeans(n_clusters=3, n_init=20, random_state=0)
        pred_ops = kmeans_ops.fit_predict(X_ops)
        # Capping silhouette at 10,000 samples to avoid O(N^2) explosion with large datasets
        metrics["silhouette_transitions"] = float(silhouette_score(X_ops, pred_ops, sample_size=10000))
    except Exception as e:
        print(f"Warning [structure_metrics]: silhouette_transitions failed: {e}")
        metrics["silhouette_transitions"] = float("nan")

    # ── Silhouette on 6-dim paper-style feature matrix ───────────────────────
    try:
        # Remove any NaN rows
        valid = ~np.isnan(transition_features).any(axis=1)
        X_feat = transition_features[valid]

        if len(X_feat) >= 4:  # silhouette_score needs at least k+1 samples
            kmeans_feat = KMeans(n_clusters=3, n_init=20, random_state=0)
            pred_feat = kmeans_feat.fit_predict(X_feat)
            metrics["silhouette_features"] = float(silhouette_score(X_feat, pred_feat, sample_size=10000))
        else:
            metrics["silhouette_features"] = float("nan")
    except Exception as e:
        print(f"Warning [structure_metrics]: silhouette_features failed: {e}")
        metrics["silhouette_features"] = float("nan")

    return {"metrics": metrics, "features": {}}
