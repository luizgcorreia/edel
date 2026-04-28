"""Visualization tools for Stage 4: Projection and Epistemic Movements."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from edel.viz.data import set_viz_style


def plot_projection_2d(
    df: pd.DataFrame,
    method: str = "umap",
    aspect: str = "problem",
    color_col: str | None = None,
    title: str | None = None,
    size: int = 20,
    alpha: float = 0.6,
    draw_arrows: bool = False,
    arrow_step: int = 20,
):
    """
    Plot the 2D epistemic landscape.
    
    Args:
        df: The projection DataFrame.
        method: DR method name (umap, pca, diffusion).
        aspect: Which aspect to plot as dots (problem, method, finding, interpretation).
        color_col: Column to use for coloring dots (e.g. cluster_id).
        draw_arrows: If True, draw p -> m -> f -> i trajectories for a subset of points.
        arrow_step: Sample every N points for arrows to avoid clutter.
    """
    set_viz_style()
    
    # Mapping for shorthand roles in legacy code vs full names in modular code
    role_map = {"p": "problem", "m": "method", "f": "finding", "i": "interpretation"}
    full_aspect = role_map.get(aspect, aspect)
    
    x_col = f"proj_{full_aspect}_{method}_x"
    y_col = f"proj_{full_aspect}_{method}_y"
    
    # Fallback for single mode
    if x_col not in df.columns:
        x_col, y_col = f"proj_{method}_x", f"proj_{method}_y"
        
    if x_col not in df.columns:
        print(f"Warning: Projection columns for {full_aspect} not found.")
        return

    plt.figure(figsize=(8, 8))
    
    if color_col and color_col in df.columns:
        sc = plt.scatter(
            df[x_col], df[y_col], 
            c=df[color_col], 
            s=size, alpha=alpha, 
            cmap="tab10" if df[color_col].dtype != float else "viridis"
        )
        plt.colorbar(sc, label=color_col)
    else:
        plt.scatter(df[x_col], df[y_col], s=size, alpha=alpha, color="#3498DB")

    if draw_arrows:
        # Check for all aspect columns
        aspects = ["problem", "method", "finding", "interpretation"]
        available = all(f"proj_{a}_{method}_x" in df.columns for a in aspects)
        if available:
            for k in range(0, len(df), arrow_step):
                # We draw thin lines connecting the stages
                pts_x = [df[f"proj_{a}_{method}_x"].iloc[k] for a in aspects]
                pts_y = [df[f"proj_{a}_{method}_y"].iloc[k] for a in aspects]
                
                # p -> m (gray)
                plt.plot(pts_x[0:2], pts_y[0:2], color="gray", alpha=0.3, lw=0.5)
                # m -> f (blue)
                plt.plot(pts_x[1:3], pts_y[1:3], color="#2980B9", alpha=0.3, lw=0.5)
                # f -> i (red)
                plt.plot(pts_x[2:4], pts_y[2:4], color="#C0392B", alpha=0.3, lw=0.5)

    plt.title(title or f"EDEL {method.upper()} Projection: {full_aspect.capitalize()}", fontsize=14, pad=15)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    plt.show()


def plot_transition_signatures(df: pd.DataFrame, bins: int = 40):
    """Plot the distribution of cosine similarities (angles) between epistemic transitions."""
    set_viz_style()
    
    # These features are usually computed during the clustering stage or manually
    # If not in DF, we might need to compute them or assume they are there
    cols = ["cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]
    
    # Check if we have the multi-aspect coordinates to compute them on the fly if missing
    if not any(c in df.columns for c in cols):
        # We can't easily compute them from projections because we need high-dim embeddings
        # for proper cosine similarity, or we compute them from projection deltas.
        # Legacy code computes them from embeddings.
        print("Warning: Transition signature columns not found in DataFrame.")
        return

    plt.figure(figsize=(12, 4))
    
    for i, col in enumerate(cols):
        if col in df.columns:
            plt.subplot(1, 3, i + 1)
            sns.histplot(df[col], bins=bins, kde=True, color="#9B59B6", alpha=0.6)
            plt.title(f"Transition: {col.replace('cos_', '')}")
            plt.xlabel("Cosine Similarity")
            plt.xlim(-1, 1)

    plt.tight_layout()
    plt.show()


def plot_movement_magnitudes(df: pd.DataFrame, bins: int = 50):
    """Plot the distribution of the magnitudes of epistemic movements."""
    set_viz_style()
    
    cols = ["mag_pm", "mag_mf", "mag_fi"]
    if not any(c in df.columns for c in cols):
        print("Warning: Magnitude columns not found.")
        return

    plt.figure(figsize=(12, 4))
    colors = ["#3498DB", "#E67E22", "#1ABC9C"]
    
    for i, col in enumerate(cols):
        if col in df.columns:
            plt.subplot(1, 3, i + 1)
            sns.histplot(df[col], bins=bins, kde=True, color=colors[i], alpha=0.6)
            plt.title(f"Magnitude: {col.replace('mag_', '')}")
            plt.xlabel("Euclidean Distance")

    plt.tight_layout()
    plt.show()


def plot_epistemic_transition_space(
    df: pd.DataFrame, 
    dimensions: int = 1536,
    quantile: float = 1.0,
    std_threshold: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    correction_method: str = "none",
    remove_pc: int = 0
):
    """
    Project the three epistemic operators (P->M, M->F, F->I) into PCA space.
    This helps visualize if the transitions are distinct in the embedding space.
    """
    from edel.pipeline.projection import load_embeddings_to_matrix
    from edel.experiments.metrics.embedding import apply_anisotropy_correction
    set_viz_style()

    aspects = ["problem", "method", "finding", "interpretation"]
    if not all(f"{a}_embedding" in df.columns for a in aspects):
        print("Error: Aspect embeddings not found in DataFrame. Cannot plot transition space.")
        return

    # 1. Load embeddings
    embs = {
        a: load_embeddings_to_matrix(df, f"{a}_embedding", dimensions)
        for a in aspects
    }

    # 1.5 Apply anisotropy correction if requested
    if correction_method != "none":
        print(f"Applying anisotropy correction ({correction_method}) to transition space...")
        embs = apply_anisotropy_correction(embs, method=correction_method, n_components=remove_pc)

    emb_p = embs["problem"]
    emb_m = embs["method"]
    emb_f = embs["finding"]
    emb_i = embs["interpretation"]

    # 2. Calculate difference vectors (Operators)
    pm = emb_m - emb_p
    mf = emb_f - emb_m
    fi = emb_i - emb_f

    # 3. Stack and Label
    X = np.vstack([pm, mf, fi])
    n = len(df)
    labels = ["P->M"] * n + ["M->F"] * n + ["F->I"] * n
    
    # 4. PCA Projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 4.5 Filter outliers if requested
    labels_arr = np.array(labels)
    mask = np.ones(len(X_pca), dtype=bool)

    if quantile < 1.0:
        # Calculate symmetric bounds based on quantile
        lower = (1.0 - quantile) / 2.0
        upper = 1.0 - lower
        
        x_min, x_max = np.percentile(X_pca[:, 0], [lower * 100, upper * 100])
        y_min, y_max = np.percentile(X_pca[:, 1], [lower * 100, upper * 100])
        
        mask &= (X_pca[:, 0] >= x_min) & (X_pca[:, 0] <= x_max) & \
                (X_pca[:, 1] >= y_min) & (X_pca[:, 1] <= y_max)

    if std_threshold is not None:
        x_mean, x_std = X_pca[:, 0].mean(), X_pca[:, 0].std()
        y_mean, y_std = X_pca[:, 1].mean(), X_pca[:, 1].std()
        
        mask &= (np.abs(X_pca[:, 0] - x_mean) <= std_threshold * x_std) & \
                (np.abs(X_pca[:, 1] - y_mean) <= std_threshold * y_std)

    if mask.sum() < len(mask):
        X_pca = X_pca[mask]
        labels_arr = labels_arr[mask]
        print(f"Filtering: Kept {mask.sum()} / {len(mask)} points.")

    # 5. Plot
    plt.figure(figsize=(10, 8))
    
    # Custom colors for the transitions
    palette = {"P->M": "#3498DB", "M->F": "#E67E22", "F->I": "#1ABC9C"}
    
    sns.scatterplot(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        hue=labels_arr, 
        palette=palette,
        alpha=0.5, 
        s=30,
        edgecolor=None
    )

    # Add centroids for clarity
    for label, color in palette.items():
        mask = labels_arr == label
        if mask.any():
            centroid = X_pca[mask].mean(axis=0)
            plt.scatter(centroid[0], centroid[1], c=color, s=200, marker='X', edgecolor='black', linewidth=2)

    plt.title(title or "Epistemic Transition Space (PCA on Operators)", fontsize=14, pad=15)
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend(title="Transition Type")
    plt.grid(True, linestyle='--', alpha=0.3)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_paper_style_pca(
    df: pd.DataFrame, 
    color_col: str | None = None, 
    quantile: float = 1.0,
    std_threshold: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None
):
    """
    Project papers into PCA space based on their 6-dimensional transition features
    (3 cosine signatures and 3 movement magnitudes).
    """
    set_viz_style()

    cols = ["cos_pm_mf", "cos_mf_fi", "cos_pm_fi", "mag_pm", "mag_mf", "mag_fi"]
    if not all(c in df.columns for c in cols):
        print("Error: Research style features not found in DataFrame. Run projection stage first.")
        return

    # 1. Prepare data
    X = df[cols].values
    X = np.nan_to_num(X) # Ensure no NaNs

    # 2. PCA Projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 2.5 Filter outliers if requested
    plot_df = pd.DataFrame({
        "x": X_pca[:, 0],
        "y": X_pca[:, 1]
    })
    if color_col and color_col in df.columns:
        plot_df[color_col] = df[color_col].values

    mask = np.ones(len(plot_df), dtype=bool)

    if quantile < 1.0:
        lower = (1.0 - quantile) / 2.0
        upper = 1.0 - lower
        
        x_min, x_max = np.percentile(X_pca[:, 0], [lower * 100, upper * 100])
        y_min, y_max = np.percentile(X_pca[:, 1], [lower * 100, upper * 100])
        
        mask &= (X_pca[:, 0] >= x_min) & (X_pca[:, 0] <= x_max) & \
                (X_pca[:, 1] >= y_min) & (X_pca[:, 1] <= y_max)

    if std_threshold is not None:
        x_mean, x_std = X_pca[:, 0].mean(), X_pca[:, 0].std()
        y_mean, y_std = X_pca[:, 1].mean(), X_pca[:, 1].std()
        
        mask &= (np.abs(X_pca[:, 0] - x_mean) <= std_threshold * x_std) & \
                (np.abs(X_pca[:, 1] - y_mean) <= std_threshold * y_std)

    if mask.sum() < len(mask):
        plot_df = plot_df[mask]
        print(f"Filtering: Kept {mask.sum()} / {len(mask)} points.")

    # 3. Plot
    plt.figure(figsize=(10, 8))
    
    if color_col and color_col in plot_df.columns:
        sc = plt.scatter(
            plot_df["x"], plot_df["y"], 
            c=plot_df[color_col], 
            s=30, alpha=0.6, 
            cmap="tab10" if plot_df[color_col].dtype != object else "tab10" 
        )
        # Note: dtype check refined for categorical/numeric
        plt.colorbar(sc, label=color_col)
    else:
        plt.scatter(plot_df["x"], plot_df["y"], s=30, alpha=0.6, color="#9B59B6")

    plt.title(title or "Paper Style Space (PCA on Transition Features)", fontsize=14, pad=15)
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.grid(True, linestyle='--', alpha=0.3)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.tight_layout()
    plt.show()
