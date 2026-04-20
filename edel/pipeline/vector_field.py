"""Stage 5: Vector Field Computation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_vector_field_stage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Orchestrate the vector field computation stage."""
    vf_cfg = config.get("vector_field", {})
    method = config.get("dimensionality_reduction", {}).get("method", "umap")
    grid_size = vf_cfg.get("grid_size", 40)
    min_count = vf_cfg.get("min_count", 0)
    compute_magnitude = vf_cfg.get("compute_magnitude", True)

    # 1. Calculate deltas (movement between epistemic aspects)
    df_with_deltas = add_deltas(df, method)

    # 2. Assign documents to grid cells based on their 'problem' projection
    df_with_grid = assign_grid(df_with_deltas, method, grid_size)

    # 3. Compute aggregated vector field
    field = compute_vector_field(df_with_grid, min_count)

    # 4. Optional: Add magnitudes
    if compute_magnitude:
        field = add_magnitudes(field)

    return field


def add_deltas(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Calculate p->m, m->f, f->i movement vectors."""
    out = df.copy()

    px, py = f"proj_problem_{method}_x", f"proj_problem_{method}_y"
    mx, my = f"proj_method_{method}_x", f"proj_method_{method}_y"
    fx, fy = f"proj_finding_{method}_x", f"proj_finding_{method}_y"
    ix, iy = f"proj_interpretation_{method}_x", f"proj_interpretation_{method}_y"

    # Check if columns exist
    for col in [px, py, mx, my, fx, fy, ix, iy]:
        if col not in out.columns:
            # If not all columns exist (e.g. single mode), we can't compute deltas
            # but we return df unchanged
            return out

    out["d_pm_x"] = out[mx] - out[px]
    out["d_pm_y"] = out[my] - out[py]

    out["d_mf_x"] = out[fx] - out[mx]
    out["d_mf_y"] = out[fy] - out[my]

    out["d_fi_x"] = out[ix] - out[fx]
    out["d_fi_y"] = out[iy] - out[fy]

    return out


def assign_grid(df: pd.DataFrame, method: str, grid_size: int) -> pd.DataFrame:
    """Map documents to grid cells based on their problem projection."""
    out = df.copy()

    px, py = f"proj_problem_{method}_x", f"proj_problem_{method}_y"
    if px not in out.columns or py not in out.columns:
        return out

    xmin, xmax = out[px].min(), out[px].max()
    ymin, ymax = out[py].min(), out[py].max()

    # Avoid division by zero
    dx = (xmax - xmin) / grid_size if xmax != xmin else 1.0
    dy = (ymax - ymin) / grid_size if ymax != ymin else 1.0

    out["cell_ix"] = np.floor((out[px] - xmin) / dx).astype(int)
    out["cell_iy"] = np.floor((out[py] - ymin) / dy).astype(int)

    out["cell_ix"] = out["cell_ix"].clip(0, grid_size - 1)
    out["cell_iy"] = out["cell_iy"].clip(0, grid_size - 1)

    out["cell_id"] = (
        out["cell_ix"].astype(str) + "_" + out["cell_iy"].astype(str)
    )

    out["_xmin"] = xmin
    out["_ymin"] = ymin
    out["_dx"] = dx
    out["_dy"] = dy

    return out


def compute_vector_field(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Group by grid cells and compute average vectors."""
    if "cell_id" not in df.columns:
        return pd.DataFrame()

    g = df.groupby("cell_id")

    field = g.agg(
        cell_ix=("cell_ix", "first"),
        cell_iy=("cell_iy", "first"),
        xmin=("_xmin", "first"),
        ymin=("_ymin", "first"),
        dx=("_dx", "first"),
        dy=("_dy", "first"),
        vf_pm_x=("d_pm_x", "mean"),
        vf_pm_y=("d_pm_y", "mean"),
        vf_mf_x=("d_mf_x", "mean"),
        vf_mf_y=("d_mf_y", "mean"),
        vf_fi_x=("d_fi_x", "mean"),
        vf_fi_y=("d_fi_y", "mean"),
        count=("cell_id", "size"),
    ).reset_index()

    field = field[field["count"] >= min_count].copy()

    # compute cell center coordinates
    field["cell_px"] = field["xmin"] + (field["cell_ix"] + 0.5) * field["dx"]
    field["cell_py"] = field["ymin"] + (field["cell_iy"] + 0.5) * field["dy"]

    return field


def add_magnitudes(field: pd.DataFrame) -> pd.DataFrame:
    """Calculate vector magnitudes for each stage."""
    if field.empty:
        return field

    field["mag_pm"] = np.sqrt(field["vf_pm_x"] ** 2 + field["vf_pm_y"] ** 2)
    field["mag_mf"] = np.sqrt(field["vf_mf_x"] ** 2 + field["vf_mf_y"] ** 2)
    field["mag_fi"] = np.sqrt(field["vf_fi_x"] ** 2 + field["vf_fi_y"] ** 2)

    return field
