"""Vector field computations for 2D overlays."""

import numpy as np


def add_vector_field_2d_smooth(
    field_df,
    field_type: str = "pm",
    xy_scale: float = 1.0,
    grid_res: int = 40,
    kernel_sigma: float = 0.3,
    step: int = 2,
    scale: float = 0.08,
):
    """Compute smoothed and normalized 2D vector arrows from cell-level vectors."""
    x_col = "cell_px"
    y_col = "cell_py"

    if field_type in ["pm", "mf", "fi"]:
        dx_col = f"vf_{field_type}_x"
        dy_col = f"vf_{field_type}_y"
        if dx_col not in field_df:
            return {"x": np.array([]), "y": np.array([]), "dx": np.array([]), "dy": np.array([])}
        DX = field_df[dx_col].values
        DY = field_df[dy_col].values
    elif field_type == "total":
        cols = ["vf_pm_x", "vf_mf_x", "vf_fi_x"]
        if not all(c in field_df for c in cols):
            return {"x": np.array([]), "y": np.array([]), "dx": np.array([]), "dy": np.array([])}
        DX = field_df["vf_pm_x"].values + field_df["vf_mf_x"].values + field_df["vf_fi_x"].values
        DY = field_df["vf_pm_y"].values + field_df["vf_mf_y"].values + field_df["vf_fi_y"].values
    elif field_type == "discovery":
        DX = field_df["vf_mf_x"].values + field_df["vf_fi_x"].values
        DY = field_df["vf_mf_y"].values + field_df["vf_fi_y"].values
    else:
        raise ValueError("Unknown field_type")

    X = field_df[x_col].values * xy_scale
    Y = field_df[y_col].values * xy_scale

    xi = np.linspace(X.min(), X.max(), grid_res)
    yi = np.linspace(Y.min(), Y.max(), grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    dx_grid = np.zeros_like(xi_grid)
    dy_grid = np.zeros_like(yi_grid)

    for px, py, vx, vy in zip(X, Y, DX, DY):
        dist2 = (xi_grid - px) ** 2 + (yi_grid - py) ** 2
        w = np.exp(-dist2 / (2 * kernel_sigma**2))
        dx_grid += w * vx
        dy_grid += w * vy

    xi_plot = xi_grid[::step, ::step]
    yi_plot = yi_grid[::step, ::step]
    dx_plot = dx_grid[::step, ::step]
    dy_plot = dy_grid[::step, ::step]

    mag = np.sqrt(dx_plot**2 + dy_plot**2)
    mask = mag > 1e-6

    xi_plot = xi_plot[mask]
    yi_plot = yi_plot[mask]
    dx_plot = dx_plot[mask]
    dy_plot = dy_plot[mask]
    mag = mag[mask]

    dx_plot /= mag
    dy_plot /= mag

    dx_plot *= scale
    dy_plot *= scale

    return {"x": xi_plot, "y": yi_plot, "dx": dx_plot, "dy": dy_plot}
