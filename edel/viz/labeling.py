"""Visualization tools for Stage 7: Labeling and Semantic Maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from edel.viz.data import set_viz_style


def print_cluster_summaries(label_results: dict, cluster_key: str = "domain"):
    """
    Print a readable summary of cluster labels and topics.
    
    Args:
        label_results: The dictionary returned by run_labeling_stage.
        cluster_key: The cluster set to summarize.
    """
    clusters = label_results.get("clusters", {}).get(cluster_key, {})
    if not clusters:
        print(f"No labels found for cluster key: {cluster_key}")
        return

    print(f"\n{'='*60}")
    print(f" CLUSTER SUMMARIES: {cluster_key.upper()}")
    print(f"{'='*60}")
    
    for cid, info in sorted(clusters.items()):
        label = info.get("proposed_label", "Unknown")
        topics = info.get("cluster_topics", "No description available.")
        print(f"\n[Cluster {cid}] -> {label.upper()}")
        print(f"Topics: {topics}")
    
    print(f"\n{'='*60}\n")


def plot_epistemic_map(
    df: pd.DataFrame,
    label_results: dict,
    method: str = "umap",
    cluster_key: str = "domain",
    topic_name: str | None = None,
    size: int = 40,
    alpha: float = 0.7,
    save_path: str | None = None,
):
    """
    Create a paper-level epistemic scatter map with semantic axis labels and cluster names.
    
    Args:
        df: DataFrame containing projections and clusters.
        label_results: The dictionary returned by run_labeling_stage.
        method: Projection method used.
        cluster_key: Cluster set to use for coloring.
        topic_name: Title prefix.
        save_path: Path to save the figure (PDF/PNG).
    """
    set_viz_style()
    sns.set_style("white") # Cleaner for paper figures
    
    cluster_col = f"cluster_{cluster_key}"
    if cluster_col not in df.columns:
        print(f"Warning: Cluster column {cluster_col} not found.")
        return

    # 1. Prepare labels for legend
    cluster_labels = label_results.get("clusters", {}).get(cluster_key, {})
    
    def get_label(cid):
        if cid == -1: return "Noise"
        info = cluster_labels.get(cid) or cluster_labels.get(str(cid))
        return info.get("proposed_label", f"Cluster {cid}") if info else f"Cluster {cid}"

    plot_df = df.copy()
    plot_df["Semantic Label"] = plot_df[cluster_col].apply(get_label)

    # 2. Prepare axis labels
    axes_info = label_results.get("axes", [])
    x_label = f"{method.upper()} 1"
    y_label = f"{method.upper()} 2"
    
    if len(axes_info) >= 1:
        x_label = axes_info[0].get("axis_label", x_label)
    if len(axes_info) >= 2:
        y_label = axes_info[1].get("axis_label", y_label)

    # 3. Plot
    plt.figure(figsize=(12, 9))
    
    # Sort by label to ensure consistent legend
    hue_order = sorted(plot_df["Semantic Label"].unique())
    if "Noise" in hue_order:
        hue_order.remove("Noise")
        hue_order.append("Noise")

    sc = sns.scatterplot(
        data=plot_df,
        x=f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x",
        y=f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df.columns else f"proj_{method}_y",
        hue="Semantic Label",
        hue_order=hue_order,
        s=size,
        alpha=alpha,
        palette="tab10" if len(hue_order) <= 10 else "tab20",
        edgecolor=None
    )

    plt.xlabel(x_label, fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")
    
    title = f"Epistemic Landscape: {topic_name}" if topic_name else "Epistemic Landscape Map"
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Knowledge Domains")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
        
    plt.show()
