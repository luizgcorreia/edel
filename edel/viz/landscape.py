"""Visualization tools for Stage 8: Interactive Epistemic Landscapes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_landscape_3d(
    df: pd.DataFrame,
    landscape_results: dict,
    method: str = "umap",
    color_col: str | None = None,
    symbol_col: str | None = None,
    title: str | None = None,
    z_label: str = "Impact",
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    label_results: dict | None = None,
    surface_opacity: float = 0.8,
    scatter_opacity: float = 0.05,
    scatter_size: int = 2,
    topic_name: str | None = None,
):
    """
    Create an interactive 3D Epistemic Landscape Surface plot.
    
    Args:
        df: DataFrame containing metadata and cluster assignments.
        landscape_results: Dictionary containing 'terrain' (xi, yi, zi).
        method: The projection method used.
        color_col: Column for dot coloring.
        symbol_col: Column for dot shapes.
    """
    terrain = landscape_results.get("terrain", {})
    xi = terrain.get("x") if "x" in terrain else terrain.get("xi")
    yi = terrain.get("y") if "y" in terrain else terrain.get("yi")
    zi = terrain.get("z") if "z" in terrain else terrain.get("zi")
    
    if xi is None or yi is None or zi is None:
        print("Warning: Terrain data missing in results (keys checked: x/xi, y/yi, z/zi).")
        return None

    # Retrieve semantic labels if available
    z_label = terrain.get("metric", z_label)
    if label_results:
        axes_info = label_results.get("axes", [])
        if len(axes_info) >= 1:
            x_label = axes_info[0].get("axis_label", x_label)
        if len(axes_info) >= 2:
            y_label = axes_info[1].get("axis_label", y_label)

    # Wrap labels for Plotly (using <br>)
    import textwrap
    x_label = "<br>".join(textwrap.wrap(x_label, width=50))
    y_label = "<br>".join(textwrap.wrap(y_label, width=50))

    fig = go.Figure()

    # 1. Add Surface
    fig.add_trace(
        go.Surface(
            x=xi, y=yi, z=zi,
            colorscale="Viridis",
            opacity=surface_opacity,
            colorbar=dict(title=z_label, x=1.05),
            name="Epistemic Surface"
        )
    )

    # 2. Add Scatter of papers (3D)
    # We need to map papers to their projection coordinates
    x_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    y_col = f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df.columns else f"proj_{method}_y"
    
    # We need the impact metric to place them at the right height
    # We use the raw_metric and log_scale flag from the terrain results
    raw_metric = terrain.get("raw_metric", "cited_by_count")
    log_scale = terrain.get("log_scale", True)
    
    if raw_metric in df.columns:
        z_vals = df[raw_metric].fillna(0).values
        if log_scale:
            z_vals = np.log10(z_vals + 1)
    else:
        z_vals = np.zeros(len(df))

    scatter = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_vals,
        color=color_col, symbol=symbol_col,
        opacity=scatter_opacity,
    )

    for trace in scatter.data:
        trace.marker.size = scatter_size
        trace.showlegend = False
        fig.add_trace(trace)

    # 3. Add Manual Legends (Workaround for PX trace merging)
    _add_manual_legends_3d(fig, df, color_col, symbol_col)

    fig.update_layout(
        title=title or f"3D Epistemic Landscape Surface: {topic_name}" if topic_name else "3D Epistemic Landscape Surface",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        width=1000, height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig


def plot_landscape_contour(
    df: pd.DataFrame,
    landscape_results: dict,
    field: pd.DataFrame | None = None,
    method: str = "umap",
    color_col: str | None = None,
    symbol_col: str | None = None,
    title: str | None = None,
    z_label: str = "Impact",
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    show_flow: bool = True,
    flow_type: str = "discovery",
    flow_scale: float = 0.20,
    topic_name: str | None = None,
    label_results: dict | None = None,
):
    """
    Create an interactive 2D Epistemic Landscape Contour Map with optional flow overlay.
    """
    terrain = landscape_results.get("terrain", {})
    xi = terrain.get("x") if "x" in terrain else terrain.get("xi")
    yi = terrain.get("y") if "y" in terrain else terrain.get("yi")
    zi = terrain.get("z") if "z" in terrain else terrain.get("zi")
    
    if xi is None or yi is None or zi is None:
        return None

    # Retrieve semantic labels if available
    z_label = terrain.get("metric", z_label)
    if label_results:
        axes_info = label_results.get("axes", [])
        if len(axes_info) >= 1:
            x_label = axes_info[0].get("axis_label", x_label)
        if len(axes_info) >= 2:
            y_label = axes_info[1].get("axis_label", y_label)

    # Wrap labels for Plotly (using <br>)
    import textwrap
    x_label = "<br>".join(textwrap.wrap(x_label, width=50))
    y_label = "<br>".join(textwrap.wrap(y_label, width=50))

    # Plotly Contour expects 1D arrays for x and y if z is a 2D grid
    x_coords = xi[0]
    y_coords = yi[:, 0]

    fig = go.Figure()

    # 1. Add Contours
    fig.add_trace(
        go.Contour(
            z=zi, x=x_coords, y=y_coords,
            colorscale="Viridis",
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title=z_label),
            opacity=0.8,
            name="Terrain Contours"
        )
    )

    # 2. Add Scatter (2D)
    x_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    y_col = f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df.columns else f"proj_{method}_y"

    scatter = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col, symbol=symbol_col,
        opacity=0.3,
    )
    
    # Add hover data if available
    hover_cols = ["title", "publication_year", "cited_by_count", "id"]
    hover_cols = [c for c in hover_cols if c in df.columns]
    if hover_cols:
        scatter.update_traces(customdata=df[hover_cols], hovertemplate="<br>".join([f"<b>{c}:</b> %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)]) + "<extra></extra>")

    for tr in scatter.data:
        tr.marker.size = 6
        tr.showlegend = False
        fig.add_trace(tr)

    # 3. Add Vector Field Overlay
    if show_flow and field is not None:
        _add_flow_to_contour(fig, field, flow_type, scale=flow_scale)

    _add_manual_legends_2d(fig, df, color_col, symbol_col)

    fig.update_layout(
        title=title or f"2D Epistemic Landscape Contour Map: {topic_name}" if topic_name else "2D Epistemic Landscape Contour Map",
        xaxis_title=x_label, yaxis_title=y_label,
        width=1000, height=800,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
    )

    return fig


# --- Private Helpers ---

def _add_manual_legends_3d(fig, df, color_col, symbol_col):
    colors = px.colors.qualitative.Set1
    symbols = ["circle", "diamond", "square", "cross", "x", "triangle-up"]
    
    if color_col and color_col in df.columns:
        labels = df[color_col].unique()
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", 
                marker=dict(size=6, color=colors[i % len(colors)]),
                legendgroup="color", legendgrouptitle_text=color_col, name=str(lab)))

    if symbol_col and symbol_col in df.columns:
        labels = df[symbol_col].unique()
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", 
                marker=dict(size=6, color="gray", symbol=symbols[i % len(symbols)]),
                legendgroup="symbol", legendgrouptitle_text=symbol_col, name=str(lab)))

def _add_manual_legends_2d(fig, df, color_col, symbol_col):
    colors = px.colors.qualitative.Set1
    symbols = ["circle", "diamond", "square", "cross", "x", "triangle-up"]
    
    if color_col and color_col in df.columns:
        labels = df[color_col].unique()
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", 
                marker=dict(size=8, color=colors[i % len(colors)]),
                legendgroup="color", legendgrouptitle_text=color_col, name=str(lab)))

    if symbol_col and symbol_col in df.columns:
        labels = df[symbol_col].unique()
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", 
                marker=dict(size=8, color="gray", symbol=symbols[i % len(symbols)]),
                legendgroup="symbol", legendgrouptitle_text=symbol_col, name=str(lab)))

def _add_flow_to_contour(fig, field, field_type, scale=0.08, grid_res=40, sigma=0.25, step=2):
    # Kernel smoothing for beautiful flow
    X = field["cell_px"].values
    Y = field["cell_py"].values
    
    if field_type == "discovery":
        DX = field["vf_mf_x"].values + field["vf_fi_x"].values
        DY = field["vf_mf_y"].values + field["vf_fi_y"].values
    else:
        DX = field.get(f"vf_{field_type}_x", np.zeros_like(X))
        DY = field.get(f"vf_{field_type}_y", np.zeros_like(Y))

    xi = np.linspace(X.min(), X.max(), grid_res)
    yi = np.linspace(Y.min(), Y.max(), grid_res)
    xi_g, yi_g = np.meshgrid(xi, yi)
    
    dx_g, dy_g = np.zeros_like(xi_g), np.zeros_like(yi_g)
    for px, py, vx, vy in zip(X, Y, DX, DY):
        dist2 = (xi_g - px)**2 + (yi_g - py)**2
        w = np.exp(-dist2 / (2 * sigma**2))
        dx_g += w * vx
        dy_g += w * vy

    # Subsample for visualization
    xs, ys = xi_g[::step, ::step].flatten(), yi_g[::step, ::step].flatten()
    dxs, dys = dx_g[::step, ::step].flatten(), dy_g[::step, ::step].flatten()
    
    # Filter small arrows
    mag = np.sqrt(dxs**2 + dys**2)
    mask = mag > 1e-5
    xs, ys, dxs, dys = xs[mask], ys[mask], dxs[mask], dys[mask]

    # Use annotations to get proper arrowheads, matching the original file's style
    # Dynamic scaling: calculate a reasonable scale based on the data range
    plot_range = X.max() - X.min()
    adjusted_scale = scale * (plot_range / 5.0) # Heuristic for good looking arrows
    
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        # Normalize and scale
        mag_i = np.sqrt(dx**2 + dy**2)
        if mag_i == 0: continue
        dx_s = (dx / mag_i) * adjusted_scale
        dy_s = (dy / mag_i) * adjusted_scale

        fig.add_annotation(
            x=x + dx_s, y=y + dy_s,
            ax=x, ay=y,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,
            arrowsize=1.2,
            arrowwidth=1,
            arrowcolor="rgba(0, 0, 0, 0.7)",
            opacity=0.7
        )
