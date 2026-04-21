"""Stage 8: Landscape - Height and smoothing calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from typing import Any, Dict, Tuple


def run_landscape_stage(
    df: pd.DataFrame, field: pd.DataFrame, config: dict
) -> Dict[str, Any]:
    """Orchestrate the landscape preparation stage."""
    ls_cfg = config.get("landscape", {})
    method = config.get("dimensionality_reduction", {}).get("method", "umap")

    # 0. Get global boundaries from documents to ensure terrain and field match perfectly
    px_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    py_col = f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df.columns else f"proj_{method}_y"
    
    if px_col in df.columns:
        x_range = (df[px_col].min(), df[px_col].max())
        y_range = (df[py_col].min(), df[py_col].max())
    else:
        x_range, y_range = None, None

    results = {}

    # 1. Terrain Calculation (Height Grid)
    z_metric = ls_cfg.get("metric", "cited_by_count")
    grid_cfg = ls_cfg.get("grid", {})
    num_bins = grid_cfg.get("num_bins", 50)
    sigma = grid_cfg.get("sigma", 1.5)
    
    log_scale = ls_cfg.get("log_scale", True)
    terrain_scale = ls_cfg.get("scale", 1.0)
    print(f"Computing terrain using metric: {z_metric} (log_scale={log_scale}, scale={terrain_scale})...")
    terrain = compute_terrain(df, method, z_metric, num_bins, sigma, log_scale, x_range, y_range, terrain_scale)
    results["terrain"] = terrain

    # 2. Smoothed Vector Field Calculation
    if not field.empty:
        kernel_sigma = ls_cfg.get("vf_kernel_sigma", 0.3)
        vf_res = ls_cfg.get("vf_resolution", 40)
        vf_type = ls_cfg.get("field", {}).get("type", "total")
        print(f"Computing smoothed vector field ({vf_type})...")
        smoothed_vf = compute_smoothed_vector_field(field, vf_res, kernel_sigma, vf_type, x_range, y_range)
        results["vector_field"] = smoothed_vf

    return results


def compute_terrain(
    df: pd.DataFrame, 
    method: str, 
    metric: str, 
    num_bins: int = 50, 
    sigma: float = 1.5,
    log_scale: bool = True,
    x_range: tuple | None = None,
    y_range: tuple | None = None,
    scale: float = 1.0
) -> Dict[str, np.ndarray]:
    """Calculate the 3D terrain grid (Z values) based on a metric."""
    
    px_col = f"proj_problem_{method}_x"
    py_col = f"proj_problem_{method}_y"
    
    if px_col not in df.columns:
        px_col, py_col = f"proj_{method}_x", f"proj_{method}_y"
        
    if px_col not in df.columns:
        return {}

    X = df[px_col].values
    Y = df[py_col].values
    
    # Handle missing metric column
    if metric not in df.columns:
        # Default to density if metric missing
        Z = np.ones_like(X)
        label = "Density"
    else:
        Z = df[metric].fillna(0).values
        label = "Citations" if metric == "cited_by_count" else metric
        
        if log_scale:
            Z = np.log10(Z + 1)
            label = f"{label} (log10)"

    # Create grid using provided ranges or data min/max
    x_min, x_max = x_range if x_range else (X.min(), X.max())
    y_min, y_max = y_range if y_range else (Y.min(), Y.max())
    
    xi = np.linspace(x_min, x_max, num_bins)
    yi = np.linspace(y_min, y_max, num_bins)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    zi_grid = np.zeros_like(xi_grid)
    
    dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
    dy = yi[1] - yi[0] if len(yi) > 1 else 1.0

    # Optimized binning
    for i in range(num_bins):
        for j in range(num_bins):
            x_min, x_max = xi[i] - dx / 2, xi[i] + dx / 2
            y_min, y_max = yi[j] - dy / 2, yi[j] + dy / 2
            
            mask = (X >= x_min) & (X < x_max) & (Y >= y_min) & (Y < y_max)
            if mask.any():
                zi_grid[j, i] = Z[mask].mean()

    # Smoothing
    if sigma > 0:
        zi_grid = gaussian_filter(zi_grid, sigma=sigma)

    # Scale Z values
    zi_grid = zi_grid * scale

    return {
        "x": xi_grid,
        "y": yi_grid,
        "z": zi_grid,
        "metric": label,
        "raw_metric": metric,
        "log_scale": log_scale
    }


def compute_smoothed_vector_field(
    field_df: pd.DataFrame, 
    grid_res: int = 40, 
    kernel_sigma: float = 0.3,
    field_type: str = "discovery",
    x_range: tuple | None = None,
    y_range: tuple | None = None
) -> Dict[str, np.ndarray]:
    """Interpolate and smooth the vector field using a Gaussian kernel."""
    
    X = field_df["cell_px"].values
    Y = field_df["cell_py"].values
    
    if field_type == "total":
        stages = ["pm", "mf", "fi"]
    elif field_type == "discovery":
        stages = ["mf", "fi"]
    else:
        stages = [field_type]

    DX = np.zeros_like(X)
    DY = np.zeros_like(Y)
    
    found_any = False
    for stage in stages:
        if f"vf_{stage}_x" in field_df.columns:
            DX += field_df[f"vf_{stage}_x"].values
            DY += field_df[f"vf_{stage}_y"].values
            found_any = True
    
    if not found_any:
        print(f"Warning: Requested field type '{field_type}' components not found in data.")

    # Create grid using provided ranges or data min/max
    x_min, x_max = x_range if x_range else (X.min(), X.max())
    y_min, y_max = y_range if y_range else (Y.min(), Y.max())
    
    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    dx_grid = np.zeros_like(xi_grid)
    dy_grid = np.zeros_like(yi_grid)

    # Kernel smoothing
    for px, py, vx, vy in zip(X, Y, DX, DY):
        dist2 = (xi_grid - px) ** 2 + (yi_grid - py) ** 2
        w = np.exp(-dist2 / (2 * kernel_sigma ** 2))
        dx_grid += w * vx
        dy_grid += w * vy

    return {
        "x": xi_grid,
        "y": yi_grid,
        "u": dx_grid,
        "v": dy_grid
    }
