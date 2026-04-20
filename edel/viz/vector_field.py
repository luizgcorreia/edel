"""Visualization tools for Stage 5: Vector Field and Epistemic Flow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from edel.viz.data import set_viz_style


def plot_vector_field(
    field: pd.DataFrame,
    field_type: str = "pm",  # "pm", "mf", "fi", "total", "discovery"
    scale: float = 1.0,
    normalize_arrows: bool = False,
    min_count: int = 3,
    color_by_mag: bool = True,
    title: str | None = None,
):
    """
    Visualize the epistemic vector field using quiver arrows.
    
    Args:
        field: The vector field DataFrame (grid cells).
        field_type: The transition to visualize.
        scale: Scale of the arrows.
        normalize_arrows: If True, all arrows have unit length (useful for direction only).
        min_count: Minimum document count per cell to display.
        color_by_mag: If True, arrows are colored by their magnitude.
    """
    set_viz_style()
    
    x = field["cell_px"].values
    y = field["cell_py"].values

    # 1. Determine vector components
    if field_type in ["pm", "mf", "fi"]:
        dx = field[f"vf_{field_type}_x"].values
        dy = field[f"vf_{field_type}_y"].values
    elif field_type == "total":
        dx = field["vf_pm_x"].values + field["vf_mf_x"].values + field["vf_fi_x"].values
        dy = field["vf_pm_y"].values + field["vf_mf_y"].values + field["vf_fi_y"].values
    elif field_type == "discovery":
        dx = field["vf_mf_x"].values + field["vf_fi_x"].values
        dy = field["vf_mf_y"].values + field["vf_fi_y"].values
    else:
        print(f"Warning: Unknown field_type '{field_type}'.")
        return

    # 2. Filter by density
    mask = field["count"].values >= min_count
    x, y, dx, dy = x[mask], y[mask], dx[mask], dy[mask]

    if len(x) == 0:
        print("Warning: No cells satisfy the min_count threshold.")
        return

    # 3. Magnitude and Normalization
    mag = np.sqrt(dx**2 + dy**2)
    if normalize_arrows:
        mag_safe = mag + 1e-8
        dx = dx / mag_safe
        dy = dy / mag_safe

    plt.figure(figsize=(10, 10))
    
    if color_by_mag:
        q = plt.quiver(
            x, y, dx, dy, mag,
            scale=scale, scale_units='xy',
            cmap="viridis", alpha=0.9
        )
        plt.colorbar(q, label="Magnitude")
    else:
        plt.quiver(
            x, y, dx, dy, 
            scale=scale, scale_units='xy',
            alpha=0.8, color="#2C3E50"
        )

    plt.title(title or f"Epistemic Vector Field Flow: {field_type.upper()}", fontsize=14, pad=15)
    plt.xlabel("Landscape Dim 1")
    plt.ylabel("Landscape Dim 2")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_field_magnitude(
    field: pd.DataFrame,
    operator: str = "pm",
    cmap: str = "magma",
    size: int = 80,
):
    """Plot a heatmap of the field magnitude across the grid."""
    set_viz_style()
    mag_col = f"mag_{operator}"
    
    if mag_col not in field.columns:
        print(f"Warning: {mag_col} not found in field.")
        return

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        field["cell_px"], field["cell_py"],
        c=field[mag_col],
        s=size,
        marker="s", # Square grid look
        cmap=cmap,
        alpha=0.9
    )
    
    plt.colorbar(sc, label="Magnitude")
    plt.title(f"Field Magnitude Intensity: {operator.upper()}", fontsize=14, pad=15)
    plt.xlabel("Landscape Dim 1")
    plt.ylabel("Landscape Dim 2")
    plt.tight_layout()
    plt.show()


def plot_field_density(
    field: pd.DataFrame,
    cmap: str = "mako",
    size: int = 80,
):
    """Plot the document density (count) across the vector field grid."""
    set_viz_style()
    
    if "count" not in field.columns:
        print("Warning: 'count' column not found.")
        return

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        field["cell_px"], field["cell_py"],
        c=field["count"],
        s=size,
        marker="s",
        cmap=cmap,
        alpha=0.9
    )
    
    plt.colorbar(sc, label="Document Count")
    plt.title("Epistemic Landscape Density (Grid)", fontsize=14, pad=15)
    plt.xlabel("Landscape Dim 1")
    plt.ylabel("Landscape Dim 2")
    plt.tight_layout()
    plt.show()
