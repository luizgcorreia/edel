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
    
    # 1.5 Compute Cluster Regions and boundaries
    print("Computing cluster regions and boundaries...")
    zi_grid = terrain.get("z", terrain.get("zi"))
    regions = compute_cluster_regions(df, method, x_range, y_range, num_bins, zi_grid=zi_grid, min_height=0.02)
    if regions:
        terrain["explored_mask"] = regions["explored_mask"]
        terrain["boundaries"] = regions["boundaries"]
        terrain["centroids"] = regions["centroids"]
        
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


def compute_cluster_regions(
    df: pd.DataFrame, 
    method: str, 
    x_range: tuple, 
    y_range: tuple, 
    num_bins: int = 50,
    min_density: int = 3,
    min_cells: int = 3,
    zi_grid: np.ndarray | None = None,
    min_height: float = 0.0
) -> Dict[str, Any]:
    """Calculate cluster boundaries and explored mask based on grid density."""
    
    px_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    py_col = f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df.columns else f"proj_{method}_y"
        
    if px_col not in df.columns or "cluster_domain" not in df.columns:
        return {}

    X = df[px_col].values
    Y = df[py_col].values
    C = df["cluster_domain"].astype(str).values
    
    x_min, x_max = x_range if x_range else (X.min(), X.max())
    y_min, y_max = y_range if y_range else (Y.min(), Y.max())
    
    xi = np.linspace(x_min, x_max, num_bins)
    yi = np.linspace(y_min, y_max, num_bins)
    
    dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
    dy = yi[1] - yi[0] if len(yi) > 1 else 1.0

    # Grid mapping
    cell_ix = np.floor((X - (x_min - dx/2)) / dx).astype(int)
    cell_iy = np.floor((Y - (y_min - dy/2)) / dy).astype(int)
    
    cell_ix = np.clip(cell_ix, 0, num_bins - 1)
    cell_iy = np.clip(cell_iy, 0, num_bins - 1)
    
    import collections
    grid_clusters = np.full((num_bins, num_bins), "", dtype=object)
    grid_density = np.zeros((num_bins, num_bins), dtype=int)
    global_density = np.zeros((num_bins, num_bins), dtype=int)
    
    # Calculate global density (all points)
    for ix, iy in zip(cell_ix, cell_iy):
        global_density[iy, ix] += 1
        
    cell_dict = collections.defaultdict(list)
    for ix, iy, c in zip(cell_ix, cell_iy, C):
        if c and c != "No topic" and c != "-1" and c != "nan":
            cell_dict[(iy, ix)].append(c)
            
    for (iy, ix), clusters in cell_dict.items():
        grid_density[iy, ix] = len(clusters)
        if len(clusters) >= min_density:
            counter = collections.Counter(clusters)
            grid_clusters[iy, ix] = counter.most_common(1)[0][0]
            
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    from scipy.ndimage import label
    
    boundaries = {}
    centroids = {}
    
    unique_clusters = [c for c in np.unique(grid_clusters) if c]
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    for cluster in unique_clusters:
        mask = (grid_clusters == cluster)
        
        # Get cell centers
        cell_x = xi_grid[mask].flatten()
        cell_y = yi_grid[mask].flatten()
        
        paths = []
        if len(cell_x) >= min_cells:
            # Centroid (center of mass of the grid cells)
            cy = yi_grid[mask].mean()
            cx = xi_grid[mask].mean()
            centroids[cluster] = {"x": cx, "y": cy}
            
            # Add 4 corners for each cell to ensure the hull encompasses the entire cell area
            pts_x = np.concatenate([cell_x - dx/2, cell_x + dx/2, cell_x - dx/2, cell_x + dx/2])
            pts_y = np.concatenate([cell_y - dy/2, cell_y - dy/2, cell_y + dy/2, cell_y + dy/2])
            pts = np.column_stack((pts_x, pts_y))
            
            try:
                hull = ConvexHull(pts)
                hx = pts[hull.vertices, 0].tolist()
                hy = pts[hull.vertices, 1].tolist()
                # Close the polygon
                hx.append(hx[0])
                hy.append(hy[0])
                paths.append({"x": hx, "y": hy})
            except Exception:
                pass # E.g., if points are perfectly collinear
                
        if paths:
            boundaries[cluster] = paths
            
    # Redefine explored mask using Global Knowledge Islands, Cluster Boundaries, and Non-zero Height
    explored_mask = np.zeros_like(xi_grid, dtype=bool)
    grid_points = np.column_stack((xi_grid.flatten(), yi_grid.flatten()))
    
    # 1. Include regions inside any cluster boundaries
    for cluster, paths in boundaries.items():
        for path_dict in paths:
            p = Path(np.column_stack((path_dict["x"], path_dict["y"])))
            inside = p.contains_points(grid_points)
            explored_mask |= inside.reshape(xi_grid.shape)
            
    # 2. Include global knowledge islands
    knowledge_threshold = 1 # Generous non-zero threshold
    global_dense_mask = global_density >= knowledge_threshold
    structure = np.ones((3, 3), dtype=int) # 8-connected
    labeled_array, num_features = label(global_dense_mask, structure=structure)
    
    for i in range(1, num_features + 1):
        island_mask = (labeled_array == i)
        island_x = xi_grid[island_mask].flatten()
        island_y = yi_grid[island_mask].flatten()
        
        if len(island_x) > 0:
            # Expand to corners
            pts_x = np.concatenate([island_x - dx/2, island_x + dx/2, island_x - dx/2, island_x + dx/2])
            pts_y = np.concatenate([island_y - dy/2, island_y - dy/2, island_y + dy/2, island_y + dy/2])
            pts = np.column_stack((pts_x, pts_y))
            
            try:
                hull = ConvexHull(pts)
                hx = pts[hull.vertices, 0].tolist()
                hy = pts[hull.vertices, 1].tolist()
                hx.append(hx[0])
                hy.append(hy[0])
                
                # Check points inside this island's hull
                p = Path(np.column_stack((hx, hy)))
                inside = p.contains_points(grid_points)
                explored_mask |= inside.reshape(xi_grid.shape)
            except Exception:
                pass

    # 3. Include any regions with height > min_height from the smoothed terrain
    if zi_grid is not None:
        explored_mask |= (zi_grid > min_height)


    
    return {
        "explored_mask": explored_mask.tolist(),
        "boundaries": boundaries,
        "centroids": centroids
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
