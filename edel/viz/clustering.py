"""Visualization tools for Stage 6: Clustering and Research Styles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from edel.viz.data import set_viz_style


def plot_clusters_on_landscape(
    df: pd.DataFrame,
    method: str = "umap",
    cluster_key: str = "domain",
    aspect: str = "problem",
    size: int = 15,
    alpha: float = 0.7,
):
    """
    Plot the documents on the landscape colored by their cluster.
    
    Args:
        df: DataFrame containing projections and clusters.
        method: Projection method (umap, pca, diffusion).
        cluster_key: The name of the cluster set (domain, style, etc.).
        aspect: The epistemic aspect to use for positions.
    """
    set_viz_style()
    cluster_col = f"cluster_{cluster_key}"
    if cluster_col not in df.columns:
        # Try fallback if not prefixed
        if cluster_key in df.columns:
            cluster_col = cluster_key
        else:
            print(f"Warning: Cluster column {cluster_col} not found.")
            return

    x_col = f"proj_{aspect}_{method}_x"
    y_col = f"proj_{aspect}_{method}_y"
    
    # Fallback for single mode
    if x_col not in df.columns:
        x_col, y_col = f"proj_{method}_x", f"proj_{method}_y"

    # Using Seaborn for better categorical legends
    unique_clusters = sorted(df[cluster_col].unique())
    cmap = "tab10" if len(unique_clusters) <= 10 else "tab20"
    
    sns.scatterplot(
        data=df, x=x_col, y=y_col,
        hue=cluster_col, palette=cmap,
        s=size, alpha=alpha,
        legend="full"
    )
    
    plt.legend(title=f"Cluster ({cluster_key})", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Clustering: {cluster_key.capitalize()} ({method.upper()})", fontsize=14, pad=15)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    plt.show()


def plot_field_clusters(
    field: pd.DataFrame,
    cluster_key: str = "field",
    size: int = 60,
):
    """
    Plot the vector field grid cells colored by their cluster.
    
    Args:
        field: The vector field DataFrame.
        cluster_key: The name of the field cluster set.
    """
    set_viz_style()
    cluster_col = f"cluster_{cluster_key}"
    if cluster_col not in field.columns:
        print(f"Warning: Field cluster column {cluster_col} not found.")
        return

    plt.figure(figsize=(8, 8))
    
    sns.scatterplot(
        data=field, x="cell_px", y="cell_py",
        hue=cluster_col, palette="tab10",
        s=size, marker="s",
        legend="full"
    )
    
    plt.legend(title=f"Field Cluster ({cluster_key})", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Vector Field Clusters: {cluster_key.capitalize()}", fontsize=14, pad=15)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.tight_layout()
    plt.show()


def plot_cluster_trajectories(
    df: pd.DataFrame,
    method: str = "umap",
    cluster_key: str = "style",
    step: int = 5,
):
    """
    Plot p -> m movement arrows colored by cluster ID.
    Especially useful for visualizing research styles/regimes.
    """
    set_viz_style()
    cluster_col = f"cluster_{cluster_key}"
    if cluster_col not in df.columns:
        print(f"Warning: Cluster column {cluster_col} not found.")
        return

    px_col, py_col = f"proj_problem_{method}_x", f"proj_problem_{method}_y"
    mx_col, my_col = f"proj_method_{method}_x", f"proj_method_{method}_y"

    if not all(c in df.columns for c in [px_col, py_col, mx_col, my_col]):
        print("Warning: Multi-aspect projections required for trajectories.")
        return

    plt.figure(figsize=(8, 8))
    
    # Draw a faint background of all points
    plt.scatter(df[px_col], df[py_col], color="gray", alpha=0.1, s=5)

    cmap = plt.cm.get_cmap("tab10")
    
    for i in range(0, len(df), step):
        c_val = df[cluster_col].iloc[i]
        if c_val == -1: continue # Skip noise
        
        plt.arrow(
            df[px_col].iloc[i],
            df[py_col].iloc[i],
            df[mx_col].iloc[i] - df[px_col].iloc[i],
            df[my_col].iloc[i] - df[py_col].iloc[i],
            color=cmap(c_val % 10),
            alpha=0.6,
            head_width=0.03,
            length_includes_head=True
        )

    plt.title(f"Research Trajectories by {cluster_key.capitalize()}", fontsize=14, pad=15)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    plt.show()
