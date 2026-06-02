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

def filter_disconnected_components(df: pd.DataFrame, embs_dict: dict, k: int, primary_aspect: str = "problem") -> Tuple[pd.DataFrame, dict, dict]:
    """
    Remove papers that belong to disconnected components in the k-NN graph.
    Returns the filtered dataframe, the filtered embeddings dict, and a report.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components
    
    if primary_aspect not in embs_dict:
        return df, embs_dict, {"n_components_before": 0, "n_dropped": 0}
        
    X = embs_dict[primary_aspect]
    print(f"Building {k}-NN graph to detect disconnected components...")
    
    A = kneighbors_graph(X, n_neighbors=k, mode='connectivity', metric='cosine', include_self=True)
    A = 0.5 * (A + A.T)
    
    n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)
    
    if n_components == 1:
        print("Graph is fully connected. No outliers dropped.")
        return df, embs_dict, {"n_components_before": 1, "n_dropped": 0}
        
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_component_label = unique_labels[np.argmax(counts)]
    
    mask = labels == largest_component_label
    n_dropped = int((~mask).sum())
    
    print(f"Detected {n_components} components. Dropping {n_dropped} outlier papers.")
    
    filtered_df = df[mask].copy().reset_index(drop=True)
    filtered_embs = {k_name: v_val[mask].copy() for k_name, v_val in embs_dict.items()}
    
    report = {
        "n_components_before": int(n_components),
        "n_dropped": n_dropped,
        "largest_component_size": int(mask.sum())
    }
    return filtered_df, filtered_embs, report


def load_embeddings_to_matrix(
    df: pd.DataFrame, column: str, dimensions: int, dtype=np.float32
) -> np.ndarray:
    """Efficiently load JSON-string embeddings into a NumPy matrix with fallback support."""
    num_rows = len(df)
    matrix = np.empty((num_rows, dimensions), dtype=dtype)

    aspects_fallback = ["problem_embedding", "method_embedding", "finding_embedding", "interpretation_embedding"]

    for i, row in enumerate(df.itertuples(index=False)):
        val = getattr(row, column, None)
        
        # If target column embedding is missing, try fallback aspects in this row
        if pd.isna(val) or val is None:
            for fallback_col in aspects_fallback:
                if fallback_col == column:
                    continue
                f_val = getattr(row, fallback_col, None)
                if not (pd.isna(f_val) or f_val is None):
                    val = f_val
                    break

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


def get_reducer(method: str, config: dict, X: np.ndarray | None = None, global_seed: int = 42) -> Any:
    """Initialize a dimensionality reduction object based on config."""
    random_state = config.get("random_state", global_seed)
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
            
        k_neighbors = config.get("diffusion_k", 30)
        adaptive_k = config.get("diffusion_adaptive_k", None)
        epsilon_cfg = config.get("diffusion_epsilon", "bgh")
        
        # 1. Compute Epsilon
        epsilon_val = epsilon_cfg
        if epsilon_cfg == "median":
            if X is None:
                logger.warning("Cannot compute median epsilon without X, falling back to 'bgh'")
                epsilon_val = "bgh"
            else:
                # Subsample to avoid memory explosion when computing pairwise distances
                from sklearn.metrics import pairwise_distances
                sample_size = min(len(X), 5000)
                indices = np.random.RandomState(random_state).choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
                # We use cosine distances
                D = pairwise_distances(X_sample, metric="cosine")
                # Median of upper triangle
                median_d = np.median(D[np.triu_indices_from(D, k=1)])
                epsilon_val = float(median_d ** 2)
                logger.info(f"Computed median epsilon for diffusion: {epsilon_val:.4f}")
                
        # 2. Adaptive Kernel (Local Scaling)
        bandwidth_fxn = None
        if adaptive_k is not None:
            class AdaptiveBandwidth:
                def __init__(self, k_bwd):
                    self.k_bwd = k_bwd
                    self.neigh = None
                    
                def __call__(self, Z):
                    from sklearn.neighbors import NearestNeighbors
                    if self.neigh is None:
                        # Fit time
                        self.neigh = NearestNeighbors(n_neighbors=self.k_bwd, metric='cosine')
                        self.neigh.fit(Z)
                        distances, _ = self.neigh.kneighbors(Z)
                        # Use MEAN distance to top k neighbors for smoother scaling
                        # This is more robust than just the k-th neighbor distance
                        avg_dist = np.mean(distances, axis=1)
                        return np.maximum(avg_dist, 1e-4)
                    else:
                        # Transform time
                        n_query = min(self.k_bwd, self.neigh.n_samples_fit_)
                        distances, _ = self.neigh.kneighbors(Z, n_neighbors=n_query)
                        avg_dist = np.mean(distances, axis=1)
                        return np.maximum(avg_dist, 1e-4)
            
            bandwidth_fxn = AdaptiveBandwidth(adaptive_k)
            logger.info(f"Using robust adaptive local scaling with k={adaptive_k}")
            
            # Ensure epsilon is at least 1.0 to avoid kernel underflow when using adaptive scaling
            if isinstance(epsilon_val, (int, float)) and epsilon_val < 1.0:
                epsilon_val = 1.0
            
        k = pydiffmap.kernel.Kernel(
            epsilon=epsilon_val, 
            metric="cosine",
            k=k_neighbors,
            bandwidth_type=bandwidth_fxn
        )
        return pydiffmap.diffusion_map.DiffusionMap(
            kernel_object=k,
            alpha=0.5,
            n_evecs=n_components,
            oos="nystroem",
        )
    else:
        raise ValueError(f"Unknown DR method: {method}")


def calculate_signatures_and_magnitudes(df: pd.DataFrame, dimensions: int) -> pd.DataFrame:
    """Calculate transition signatures (cosines) and movement magnitudes between aspects."""
    aspects = ["problem", "method", "finding", "interpretation"]
    if not all(f"{a}_embedding" in df.columns for a in aspects):
        return df

    # Load embeddings to matrices
    emb_p = load_embeddings_to_matrix(df, "problem_embedding", dimensions)
    emb_m = load_embeddings_to_matrix(df, "method_embedding", dimensions)
    emb_f = load_embeddings_to_matrix(df, "finding_embedding", dimensions)
    emb_i = load_embeddings_to_matrix(df, "interpretation_embedding", dimensions)

    # Calculate difference vectors (Operators)
    pm = emb_m - emb_p
    mf = emb_f - emb_m
    fi = emb_i - emb_f

    # Calculate Magnitudes
    df["mag_pm"] = np.linalg.norm(pm, axis=1)
    df["mag_mf"] = np.linalg.norm(mf, axis=1)
    df["mag_fi"] = np.linalg.norm(fi, axis=1)

    # Calculate Transition Signatures (Cosine similarities between operators)
    def row_wise_cosine(a, b):
        # Normalize rows to unit length
        a_norm = normalize(a, axis=1, norm="l2")
        b_norm = normalize(b, axis=1, norm="l2")
        # Dot product of normalized rows = cosine similarity
        return np.sum(a_norm * b_norm, axis=1)

    df["cos_pm_mf"] = row_wise_cosine(pm, mf)
    df["cos_mf_fi"] = row_wise_cosine(mf, fi)
    df["cos_pm_fi"] = row_wise_cosine(pm, fi)

    return df


def run_projection_stage(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """Orchestrate the dimensionality reduction stage."""
    dr_cfg = config.get("dimensionality_reduction", {})
    method = dr_cfg.get("method", "umap")
    remove_pc = dr_cfg.get("remove_top_pcs", 0)
    anisotropy_method = dr_cfg.get("anisotropy_method", "pc_removal" if remove_pc > 0 else "none")
    dimensions = config.get("embedding", {}).get("n_dimensions", 1536)
    
    out = df.copy()
    report = {}

    # Determine which columns to project
    # Support both 'single' (embedding) and 'multi' (problem_embedding, etc.)
    if "embedding" in out.columns:
        # Single mode
        print(f"Projecting 'embedding' column using {method}...")
        X = load_embeddings_to_matrix(out, "embedding", dimensions)
        
        if anisotropy_method != "none":
            from edel.experiments.metrics.embedding import apply_anisotropy_correction
            print(f"Applying anisotropy correction ({anisotropy_method})...")
            X = apply_anisotropy_correction({"embedding": X}, method=anisotropy_method, n_components=remove_pc)["embedding"]
            
        if method == "diffusion" and dr_cfg.get("filter_disconnected", False):
            k = dr_cfg.get("diffusion_k", 100)
            out, temp_dict, filt_report = filter_disconnected_components(out, {"embedding": X}, k=k, primary_aspect="embedding")
            X = temp_dict["embedding"]
            report["filtering"] = filt_report
            
        X_prep = prepare_matrix(X)
        
        seed = config.get("random_seed", 42)
        reducer = get_reducer(method, dr_cfg, X=X_prep, global_seed=seed)
        coords = reducer.fit_transform(X_prep)
        
        if method == "diffusion" and hasattr(reducer, "evals"):
            report["evals"] = reducer.evals.tolist()
            
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
        
        # Load and optionally denoise
        embs_to_proj = {
            a: load_embeddings_to_matrix(out, f"{a}_embedding", dimensions)
            for a in available_aspects
        }
        
        if anisotropy_method != "none":
            from edel.experiments.metrics.embedding import apply_anisotropy_correction
            print(f"Applying anisotropy correction ({anisotropy_method})...")
            embs_to_proj = apply_anisotropy_correction(embs_to_proj, method=anisotropy_method, n_components=remove_pc)

        if method == "diffusion" and dr_cfg.get("filter_disconnected", False):
            k = dr_cfg.get("diffusion_k", 100)
            out, embs_to_proj, filt_report = filter_disconnected_components(out, embs_to_proj, k=k, primary_aspect="problem")
            report["filtering"] = filt_report

        # 1. Fit on 'problem' (legacy logic: problem defines the coordinate system)
        primary_aspect = "problem"
        X_primary = embs_to_proj[primary_aspect]
        X_primary_prep = prepare_matrix(X_primary)
        
        seed = config.get("random_seed", 42)
        reducer = get_reducer(method, dr_cfg, X=X_primary_prep, global_seed=seed)
        coords_primary = reducer.fit_transform(X_primary_prep)
        
        if method == "diffusion" and hasattr(reducer, "evals"):
            report["evals"] = reducer.evals.tolist()
            
        # Save primary results
        out[f"proj_{primary_aspect}_{method}_x"] = coords_primary[:, 0]
        out[f"proj_{primary_aspect}_{method}_y"] = coords_primary[:, 1]
        
        # 2. Transform other aspects using the SAME reducer
        for aspect in available_aspects:
            if aspect == primary_aspect:
                continue
                
            print(f"Transforming {aspect} aspect...")
            X_a = embs_to_proj[aspect]
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

        # 3. Calculate Signatures & Magnitudes (Research Style Features)
        print("Calculating transition signatures and magnitudes...")
        out = calculate_signatures_and_magnitudes(out, dimensions)

    # Memory efficiency: Drop the large embedding columns if requested
    if dr_cfg.get("drop_embeddings", False):
        cols_to_drop = [c for c in out.columns if "embedding" in c]
        print(f"Dropping {len(cols_to_drop)} embedding columns to save memory.")
        out = out.drop(columns=cols_to_drop)

    return out, report
