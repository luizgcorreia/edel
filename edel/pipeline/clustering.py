"""Stage 6: Clustering"""

from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

try:
    import hdbscan
except ImportError:
    hdbscan = None

from edel.pipeline.projection import load_embeddings_to_matrix

logger = logging.getLogger(__name__)


def run_clustering_stage(
    df: pd.DataFrame, field: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrate the clustering stage."""
    cluster_cfg = config.get("clustering", {})
    dimensions = config.get("embedding", {}).get("n_dimensions", 1536)

    out_df = df.copy()
    out_field = field.copy()

    for name, cfg in cluster_cfg.items():
        source = cfg.get("source", "proj_p")
        algorithm = cfg.get("algorithm", "kmeans")
        params = cfg.get("params", {})

        print(f"Running clustering: {name} (source: {source}, algorithm: {algorithm})...")

        try:
            X = get_clustering_matrix(source, out_df, out_field, dimensions)
            if X is None or len(X) == 0:
                print(f"Skipping clustering {name}: No data found for source {source}")
                continue

            seed = config.get("random_seed", 42)
            labels = run_clustering(X, algorithm, params, random_seed=seed)

            colname = f"cluster_{name}"
            if source == "field":
                out_field[colname] = labels
            else:
                out_df[colname] = labels

            # Optional: Log silhouette score if possible
            if len(np.unique(labels)) > 1 and len(labels) > 2:
                try:
                    score = silhouette_score(X, labels)
                    print(f"Silhouette Score for {name}: {score:.4f}")
                except Exception:
                    pass

        except Exception as e:
            print(f"Error during clustering {name}: {e}")

    return out_df, out_field


def get_clustering_matrix(
    source: str, df: pd.DataFrame, field: pd.DataFrame, dimensions: int
) -> Optional[np.ndarray]:
    """Extract and prepare the data matrix for clustering based on the source."""
    
    if source == "emb_p":
        if "problem_embedding" not in df.columns:
            return None
        return load_embeddings_to_matrix(df, "problem_embedding", dimensions)

    elif source == "proj_p":
        # Look for the first projection found (e.g. proj_problem_umap_x)
        cols = [c for c in df.columns if c.startswith("proj_problem_") and (c.endswith("_x") or c.endswith("_y"))]
        if not cols:
            # Fallback for single mode
            cols = [c for c in df.columns if c.startswith("proj_") and (c.endswith("_x") or c.endswith("_y"))]
        
        if not cols:
            return None
        return df[cols].values

    elif source == "features":
        # Hand-crafted transition features
        return compute_transition_features(df, dimensions)

    elif source == "operators":
        # Stacked deltas
        deltas = ["d_pm_x", "d_pm_y", "d_mf_x", "d_mf_y", "d_fi_x", "d_fi_y"]
        if not all(d in df.columns for d in deltas):
            return None
        return df[deltas].values

    elif source == "field":
        if field.empty:
            return None
        cols = [c for c in field.columns if c.startswith("vf_") or c.startswith("mag_")]
        if not cols:
            return None
        return field[cols].values

    else:
        raise ValueError(f"Unknown clustering source: {source}")


def compute_transition_features(df: pd.DataFrame, dimensions: int) -> np.ndarray:
    """Compute cosine-based transition features between epistemic aspects."""
    aspects = ["problem", "method", "finding", "interpretation"]
    if not all(f"{a}_embedding" in df.columns for a in aspects):
        raise ValueError("Multi-aspect embeddings required for transition features")

    emb_p = load_embeddings_to_matrix(df, "problem_embedding", dimensions)
    emb_m = load_embeddings_to_matrix(df, "method_embedding", dimensions)
    emb_f = load_embeddings_to_matrix(df, "finding_embedding", dimensions)
    emb_i = load_embeddings_to_matrix(df, "interpretation_embedding", dimensions)

    d_pm = emb_m - emb_p
    d_mf = emb_f - emb_m
    d_fi = emb_i - emb_f

    pm_n = normalize(d_pm, axis=1)
    mf_n = normalize(d_mf, axis=1)
    fi_n = normalize(d_fi, axis=1)

    def cos(a, b):
        return np.sum(a * b, axis=1)

    features = np.column_stack(
        [
            cos(pm_n, mf_n),
            cos(pm_n, fi_n),
            cos(mf_n, fi_n),
            np.linalg.norm(d_pm, axis=1),
            np.linalg.norm(d_mf, axis=1),
            np.linalg.norm(d_fi, axis=1),
        ]
    )

    return features


def run_clustering(X: np.ndarray, algorithm: str, params: dict, random_seed: int = 42) -> np.ndarray:
    """Run the specified clustering algorithm on the matrix X."""
    params = params.copy()
    # Centering and Normalizing
    X_norm = X - np.mean(X, axis=0)
    X_norm = normalize(X_norm, axis=1)

    if algorithm == "kmeans":
        if "random_state" not in params:
            params["random_state"] = random_seed
        model = KMeans(**params)
        return model.fit_predict(X_norm)
    elif algorithm == "hdbscan":
        if hdbscan is None:
            raise ImportError("hdbscan is not installed")
        model = hdbscan.HDBSCAN(**params)
        return model.fit_predict(X_norm)
    elif algorithm == "gmm":
        if "random_state" not in params:
            params["random_state"] = random_seed
        model = GaussianMixture(**params)
        return model.fit(X_norm).predict(X_norm)
    elif algorithm == "spectral":
        model = SpectralClustering(**params)
        return model.fit_predict(X_norm)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(**params)
        return model.fit_predict(X_norm)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
