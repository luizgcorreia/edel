"""Visualization tools for Stage 1: Data Collection & Distribution Analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_viz_style():
    """Apply a premium styling to matplotlib/seaborn plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.family"] = "sans-serif"


def plot_abstract_length_dist(
    df: pd.DataFrame, max_len: int = 600, color: str = "#4A90E2"
):
    """Plot the distribution of abstract lengths (in tokens/words)."""
    set_viz_style()
    if "abstract_text" not in df.columns:
        print("Warning: 'abstract_text' column not found.")
        return

    # Calculate length if not already present in a cached column
    lengths = df["abstract_text"].str.split().str.len()

    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50, kde=True, color=color, alpha=0.7)
    
    plt.title("Distribution of Abstract Text Lengths", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Abstract Length (words)", fontsize=12)
    plt.ylabel("Number of Works", fontsize=12)
    plt.xlim(0, max_len)
    plt.tight_layout()
    plt.show()


def plot_publication_year_dist(
    df: pd.DataFrame, min_year: int = 1940, max_year: int = 2025, color: str = "#E67E22"
):
    """Plot the distribution of publication years."""
    set_viz_style()
    if "publication_year" not in df.columns:
        print("Warning: 'publication_year' column not found.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df["publication_year"], bins=50, kde=True, color=color, alpha=0.7)
    
    plt.title("Distribution of Publication Years", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Publication Year", fontsize=12)
    plt.ylabel("Number of Works", fontsize=12)
    plt.xlim(min_year, max_year)
    plt.tight_layout()
    plt.show()


def plot_citation_dist(
    df: pd.DataFrame, max_citations: int | None = None, color: str = "#16A085"
):
    """Plot the distribution of citation counts."""
    set_viz_style()
    if "cited_by_count" not in df.columns:
        print("Warning: 'cited_by_count' column not found.")
        return

    plt.figure(figsize=(10, 6))
    
    # Filter out NaNs for plotting
    data = df["cited_by_count"].dropna()
    
    if data.empty:
        print("Warning: No citation data available to plot.")
        return

    # Using more bins for citations since it's often a power-law distribution
    sns.histplot(data, bins=min(len(data), 50), kde=True, color=color, alpha=0.7)
    
    plt.title("Distribution of Citations", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Number of Citations", fontsize=12)
    plt.ylabel("Number of Works", fontsize=12)
    
    # If max_citations is provided, cap the plot; otherwise use data max
    if max_citations:
        plt.xlim(0, max_citations)
    
    plt.tight_layout()
    plt.show()


def plot_segmentation_stats(report: dict):
    """Plot metrics from the structured abstract segmentation step."""
    set_viz_style()
    
    # 1. Lengths
    aspect_order = ["problem", "method", "finding", "interpretation"]
    len_keys = [f"len_{a}" for a in aspect_order if f"len_{a}" in report]
    
    if len_keys:
        plt.figure(figsize=(10, 5))
        labels = [k.replace("len_", "").capitalize() for k in len_keys]
        values = [report[k] for k in len_keys]
        
        sns.barplot(x=labels, y=values, palette="mako")
        plt.title("Average Segment Length (tokens)", fontsize=14, fontweight="bold", pad=20)
        plt.ylabel("Avg. Tokens")
        
        # Add text labels on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")
            
        plt.tight_layout()

    # 2. Coverage Ratio
    if "seg_ratio_mean" in report:
        plt.figure(figsize=(8, 4))
        ratio = report["seg_ratio_mean"]
        
        # Color based on coverage (too high > 1.2 or too low < 0.6 might be suspicious)
        color = "#16A085" if 0.7 <= ratio <= 1.1 else "#E67E22"
        
        plt.barh(["Abstract Coverage"], [ratio], color=color, alpha=0.8)
        plt.axvline(1.0, color="#C0392B", linestyle="--", alpha=0.6, label="Original Abstract Length")
        
        plt.title(f"Segmentation Coverage Ratio: {ratio:.1%}", fontsize=14, fontweight="bold", pad=20)
        plt.xlabel("Total Segments Length / Abstract Length")
        plt.legend(loc="lower right")
        plt.xlim(0, max(1.3, ratio + 0.2))
        plt.tight_layout()


def plot_filtering_stats(report: dict):
    """Plot keyword filtering results from data collection."""
    set_viz_style()
    
    if "keyword_stats" not in report or not report["keyword_stats"]:
        return

    stats = report["keyword_stats"]
    # Sort by count descending, only include non-zero
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    keywords = [x[0] for x in sorted_stats if x[1] > 0]
    counts = [x[1] for x in sorted_stats if x[1] > 0]

    if not counts:
        # If no keywords matched but items were removed (e.g. is_paratext:true)
        # show a simple summary if removed_count > 0
        if report.get("removed_count", 0) > 0:
            plt.figure(figsize=(8, 2))
            plt.barh(["Items Removed"], [report["removed_count"]], color="#C0392B")
            plt.title("Data Collection Filtering Summary", fontsize=14, fontweight="bold")
            plt.tight_layout()
        return

    plt.figure(figsize=(10, min(max(4, len(keywords) * 0.5), 10)))
    sns.barplot(x=counts, y=keywords, palette="flare")
    
    plt.title(f"Non-Research Item Matches (Total Removed: {report.get('removed_count', 0)})", 
              fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Number of Matches", fontsize=12)
    plt.ylabel("Filter Keyword", fontsize=12)
    
    # Add text labels
    for i, v in enumerate(counts):
        plt.text(v + (max(counts) * 0.01), i, str(v), va="center", fontweight="bold")
        
    plt.tight_layout()
