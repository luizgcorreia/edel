"""Visualization tools for Stage 4: Projection and Epistemic Movements."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
