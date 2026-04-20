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
    df: pd.DataFrame, max_citations: int = 50, color: str = "#16A085"
):
    """Plot the distribution of citation counts."""
    set_viz_style()
    if "cited_by_count" not in df.columns:
        print("Warning: 'cited_by_count' column not found.")
        return

    plt.figure(figsize=(10, 6))
    # Using more bins for citations since it's often a power-law distribution
    sns.histplot(df["cited_by_count"], bins=100, kde=True, color=color, alpha=0.7)
    
    plt.title("Distribution of Citations", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Number of Citations", fontsize=12)
    plt.ylabel("Number of Works", fontsize=12)
    plt.xlim(0, max_citations)
    # Optional: set y-limit if there are too many 0-citation papers
    # plt.ylim(0, df['cited_by_count'].value_counts().max() * 1.1) 
    plt.tight_layout()
    plt.show()
