"""Stage 4: Dimensionality Reduction."""

from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import umap

try:
    import pydiffmap
except ImportError:
    pydiffmap = None

logger = logging.getLogger(__name__)


def load_embeddings_to_matrix(
    df: pd.DataFrame, column: str, dimensions: int, dtype=np.float32
) -> np.ndarray:
    """Efficiently load JSON-string embeddings into a NumPy matrix."""
    num_rows = len(df)
    matrix = np.empty((num_rows, dimensions), dtype=dtype)

    for i, val in enumerate(df[column]):
        if pd.isna(val) or val is None:
            matrix[i, :] = 0.0
            continue
        try:
            if isinstance(val, str):
                parsed = json.loads(val)
            else:
                parsed = val

            if parsed is None:
                matrix[i, :] = 0.0
            elif isinstance(parsed, list):
                if len(parsed) != dimensions:
                    logger.warning(
                        f"Row {i} in '{column}' has length {len(parsed)}, expected {dimensions}."
                    )
                    matrix[i, :] = 0.0
                else:
                    matrix[i, :] = parsed
            else:
                matrix[i, :] = 0.0
        except Exception as e:
            logger.error(f"Error parsing embedding at row {i}: {e}")
            matrix[i, :] = 0.0

    return matrix


def prepare_matrix(X: np.ndarray) -> np.ndarray:
    """Legacy preprocessing: centering and normalizing."""
    # Mean centering
    X_centered = X - np.mean(X, axis=0)
    # Unit normalization (cosine similarity preservation)
    X_norm = normalize(X_centered, axis=1, norm="l2")
    return X_norm


def get_reducer(method: str, config: dict) -> Any:
    """Initialize a dimensionality reduction object based on config."""
    random_state = config.get("random_state", 42)
    n_components = config.get("n_components", 2)

    if method == "umap":
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=config.get("n_neighbors", 15),
            min_dist=config.get("min_dist", 0.1),
            metric="cosine",
            random_state=random_state,
        )
    elif method == "pca":
        return PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        return TSNE(n_components=n_components, random_state=random_state)
    elif method == "diffusion":
        if pydiffmap is None:
            raise ImportError("pydiffmap is not installed")
        k = pydiffmap.kernel.Kernel(epsilon=1, metric="cosine")
        return pydiffmap.diffusion_map.DiffusionMap(
            kernel_object=k,
            alpha=0.5,
            n_evecs=n_components,
            oos="nystroem",
        )
    else:
        raise ValueError(f"Unknown DR method: {method}")


def run_projection_stage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Orchestrate the dimensionality reduction stage."""
    dr_cfg = config.get("dimensionality_reduction", {})
    method = dr_cfg.get("method", "umap")
    dimensions = config.get("embedding", {}).get("n_dimensions", 1536)
    
    out = df.copy()

    # Determine which columns to project
    # Support both 'single' (embedding) and 'multi' (problem_embedding, etc.)
    if "embedding" in out.columns:
        # Single mode
        print(f"Projecting 'embedding' column using {method}...")
        X = load_embeddings_to_matrix(out, "embedding", dimensions)
        X_prep = prepare_matrix(X)
        
        reducer = get_reducer(method, dr_cfg)
        coords = reducer.fit_transform(X_prep)
        
        out[f"proj_{method}_x"] = coords[:, 0]
        out[f"proj_{method}_y"] = coords[:, 1]
        if coords.shape[1] > 2:
            out[f"proj_{method}_z"] = coords[:, 2]
            
    elif "problem_embedding" in out.columns:
        # Multi mode (Epistemic Aspects)
        aspects = ["problem", "method", "finding", "interpretation"]
        available_aspects = [a for a in aspects if f"{a}_embedding" in out.columns]
        
        if not available_aspects:
            print("No aspect embeddings found to project.")
            return out
            
        print(f"Projecting {len(available_aspects)} aspects into common {method} space...")
        
        # 1. Fit on 'problem' (legacy logic: problem defines the coordinate system)
        primary_aspect = "problem"
        X_primary = load_embeddings_to_matrix(out, f"{primary_aspect}_embedding", dimensions)
        X_primary_prep = prepare_matrix(X_primary)
        
        reducer = get_reducer(method, dr_cfg)
        coords_primary = reducer.fit_transform(X_primary_prep)
        
        # Save primary results
        out[f"proj_{primary_aspect}_{method}_x"] = coords_primary[:, 0]
        out[f"proj_{primary_aspect}_{method}_y"] = coords_primary[:, 1]
        
        # 2. Transform other aspects using the SAME reducer
        for aspect in available_aspects:
            if aspect == primary_aspect:
                continue
                
            print(f"Transforming {aspect} aspect...")
            X_a = load_embeddings_to_matrix(out, f"{aspect}_embedding", dimensions)
            X_a_prep = prepare_matrix(X_a)
            
            # transform() might not be available for all methods (e.g. t-SNE)
            try:
                coords_a = reducer.transform(X_a_prep)
                out[f"proj_{aspect}_{method}_x"] = coords_a[:, 0]
                out[f"proj_{aspect}_{method}_y"] = coords_a[:, 1]
            except AttributeError:
                print(f"Warning: {method} does not support transform(). Refitting for {aspect}...")
                coords_a = reducer.fit_transform(X_a_prep)
                out[f"proj_{aspect}_{method}_x"] = coords_a[:, 0]
                out[f"proj_{aspect}_{method}_y"] = coords_a[:, 1]

    # Memory efficiency: Drop the large embedding columns if requested
    if dr_cfg.get("drop_embeddings", False):
        cols_to_drop = [c for c in out.columns if "embedding" in c]
        print(f"Dropping {len(cols_to_drop)} embedding columns to save memory.")
        out = out.drop(columns=cols_to_drop)

    return out
