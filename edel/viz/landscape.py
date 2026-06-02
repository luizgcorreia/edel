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
    show_papers: bool = False,
    top_papers_n: int = 10,
    papers_metric: str = "cited_by_count",
    max_scatter_points: int = 1000,
    random_seed: int = 42,
):
    """
    Create an interactive 3D Epistemic Landscape Surface plot.
    
    Args:
        df: DataFrame containing metadata and cluster assignments.
        landscape_results: Dictionary containing 'terrain' (xi, yi, zi).
        method: The projection method used.
        color_col: Column for dot coloring.
        symbol_col: Column for dot shapes.
        max_scatter_points: Maximum number of papers to include in the scatter layer.
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
    # Ensure categorical color column is string to avoid double colorbars
    df_plot = df.copy()
    if color_col and color_col in df_plot.columns:
        df_plot[color_col] = df_plot[color_col].astype(str)

    # Sampling for large datasets
    if len(df_plot) > max_scatter_points:
        print(f"Sampling {max_scatter_points} points for scatter plot (original: {len(df_plot)})...")
        df_scatter = df_plot.sample(n=max_scatter_points, random_state=random_seed)
    else:
        df_scatter = df_plot

    # Resolve projection method fallback if missing
    x_cols = [c for c in df.columns if c.startswith("proj_") and c.endswith("_x")]
    requested_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    if requested_col not in df.columns and x_cols:
        fallback_col = x_cols[0]
        if fallback_col.startswith("proj_problem_"):
            method = fallback_col[len("proj_problem_"):-2]
        else:
            method = fallback_col[len("proj_"):-2]

    # We need to map papers to their projection coordinates
    x_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    y_col = f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df.columns else f"proj_{method}_y"
    
    # We need the impact metric to place them at the right height
    # We use the raw_metric and log_scale flag from the terrain results
    raw_metric = terrain.get("raw_metric", "cited_by_count")
    log_scale = terrain.get("log_scale", True)
    
    if raw_metric in df_scatter.columns:
        z_vals = df_scatter[raw_metric].fillna(0).values
        if log_scale:
            z_vals = np.log10(z_vals + 1)
    else:
        z_vals = np.zeros(len(df_scatter))

    # Add hover data if available (Clean hover, detailed customdata)
    hover_cols = ["title", "publication_year", "cited_by_count", "id", "problem", "method", "finding", "interpretation", "doi"]
    hover_cols = [c for c in hover_cols if c in df_scatter.columns]
    
    # Just show these on hover
    display_hover = ["title", "publication_year", "cited_by_count"]
    display_hover = [c for c in display_hover if c in df_scatter.columns]

    scatter = px.scatter_3d(
        df_scatter, x=x_col, y=y_col, z=z_vals,
        color=color_col, symbol=symbol_col,
        opacity=scatter_opacity,
        color_discrete_sequence=px.colors.qualitative.Set1,
        hover_data=hover_cols
    )

    for trace in scatter.data:
        trace.marker.size = scatter_size
        trace.showlegend = False
        if hover_cols:
            # Use only a subset for the hover box to keep it readable
            trace.hovertemplate = "<br>".join([f"<b>{c}:</b> %{{customdata[{hover_cols.index(c)}]}}" for c in display_hover]) + "<extra></extra>"
        fig.add_trace(trace)
        
    # 2.5 Add Relevant Papers ("Cities")
    if show_papers:
        import datetime
        from edel.dashboard.utils import parse_authorships
        
        df_papers = df_plot.copy()
        
        if papers_metric == "cited_by_count":
            df_papers = df_papers.sort_values(by="cited_by_count", ascending=False)
            top_papers = df_papers.head(top_papers_n)
        elif papers_metric == "citation_velocity":
            current_year = datetime.datetime.now().year
            years_since = np.maximum(1, current_year - df_papers["publication_year"].fillna(current_year))
            df_papers["velocity"] = df_papers["cited_by_count"].fillna(0) / years_since
            df_papers = df_papers.sort_values(by="velocity", ascending=False)
            top_papers = df_papers.head(top_papers_n)
        elif papers_metric == "citation_normalized_percentile":
            if "citation_normalized_percentile" in df_papers.columns:
                df_papers = df_papers.sort_values(by="citation_normalized_percentile", ascending=False)
            else:
                df_papers = df_papers.sort_values(by="cited_by_count", ascending=False)
            top_papers = df_papers.head(top_papers_n)
        elif papers_metric == "cluster_centroids":
            if color_col and color_col in df_papers.columns:
                clusters = df_papers[color_col].unique()
                k = max(1, top_papers_n // max(1, len(clusters)))
                top_papers = df_papers.sort_values("cited_by_count", ascending=False).groupby(color_col).head(k).head(top_papers_n)
            else:
                top_papers = df_papers.sort_values(by="cited_by_count", ascending=False).head(top_papers_n)
        elif papers_metric == "local_peaks":
            if zi is not None:
                x_min = xi.min()
                y_min = yi.min()
                if len(xi.shape) > 1:
                    dx = xi[0,1] - xi[0,0] if xi.shape[1] > 1 else 1.0
                else:
                    dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
                    
                if len(yi.shape) > 1:
                    dy = yi[1,0] - yi[0,0] if yi.shape[0] > 1 else 1.0
                else:
                    dy = yi[1] - yi[0] if len(yi) > 1 else 1.0
                c_ix = np.clip(np.floor((df_papers[x_col] - (x_min - dx/2)) / dx).astype(int), 0, zi.shape[1]-1)
                c_iy = np.clip(np.floor((df_papers[y_col] - (y_min - dy/2)) / dy).astype(int), 0, zi.shape[0]-1)
                df_papers["peak_height"] = zi[c_iy, c_ix]
                df_papers = df_papers.sort_values(["peak_height", "cited_by_count"], ascending=[False, False])
                top_papers = df_papers.head(top_papers_n)
            else:
                top_papers = df_papers.sort_values(by="cited_by_count", ascending=False).head(top_papers_n)
        else:
            top_papers = df_papers.sort_values(by="cited_by_count", ascending=False).head(top_papers_n)

        labels = []
        for _, row in top_papers.iterrows():
            authors_list = parse_authorships(row.get("authorships", ""))
            first_author = authors_list[0].get("name", "Unknown") if authors_list else "Unknown"
            last_name = first_author.split()[-1] if first_author != "Unknown" else "Unknown"
            year = str(row.get("publication_year", ""))[:4]
            labels.append(f"{last_name} ({year})")
            
        # Re-calculate Z for these specific points to ensure they sit on the surface
        if raw_metric in top_papers.columns:
            tp_z_vals = top_papers[raw_metric].fillna(0).values
            if log_scale:
                tp_z_vals = np.log10(tp_z_vals + 1)
        else:
            tp_z_vals = np.zeros(len(top_papers))
            
        fig.add_trace(go.Scatter3d(
            x=top_papers[x_col],
            y=top_papers[y_col],
            z=tp_z_vals,
            mode='markers+text',
            text=labels,
            textposition="top center",
            textfont=dict(color="white", size=11, family="Arial Black"),
            name="Relevant Papers",
            showlegend=False,
            hoverinfo='skip'
        ))

    # 3. Add Manual Legends (Workaround for PX trace merging)
    _add_manual_legends_3d(fig, df_plot, color_col, symbol_col, label_results)

    fig.update_layout(
        title=title or f"3D Epistemic Landscape Surface: {topic_name}" if topic_name else "3D Epistemic Landscape Surface",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        width=1280, height=800, # 16:10 aspect ratio
        margin=dict(l=50, r=50, b=50, t=50),
        autosize=False,
        legend=dict(
            x=0.98, y=0.98, 
            xanchor="right", yanchor="top", 
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    # 4. Add Pre-allocated Trajectory Placeholders (for callbacks)
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines+markers", 
                               line=dict(color="red", width=5), marker=dict(size=4, color="red"),
                               name="Trajectory Line", showlegend=False, hoverinfo="skip", visible=False))
                               
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", 
                               marker=dict(size=8, color="yellow", line=dict(width=2, color="red")), 
                               name="Selected Paper", showlegend=False, hoverinfo="skip", visible=False))

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
    flow_width: float = 0.7,
    arrow_size: float = 0.8,
    topic_name: str | None = None,
    label_results: dict | None = None,
    show_regions: bool = False,
    show_frontier: bool = False,
    show_papers: bool = False,
    top_papers_n: int = 10,
    papers_metric: str = "cited_by_count",
    max_scatter_points: int = 1000,
    random_seed: int = 42,
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

    # Apply explored mask if frontier is enabled
    explored_mask = terrain.get("explored_mask")
    if show_frontier and explored_mask is not None:
        # Convert mask to boolean array
        mask_arr = np.array(explored_mask, dtype=bool)
        if mask_arr.shape == zi.shape:
            # Copy to avoid modifying the original artifact in memory
            zi = np.copy(zi)
            zi[~mask_arr] = np.nan
            
    # 1. Add Contours
    fig.add_trace(
        go.Contour(
            z=zi, x=x_coords, y=y_coords,
            colorscale="Viridis",
            contours=dict(showlabels=False),
            colorbar=dict(title=z_label),
            opacity=0.8,
            name="Terrain Contours"
        )
    )
    
    if show_frontier and explored_mask is not None:
        fig.update_layout(plot_bgcolor="rgba(60, 60, 60, 1)") # Distinct gray for unexplored
    else:
        fig.update_layout(plot_bgcolor="#440154") # Lowest Viridis color for seamless background

    # 2. Add Scatter (2D)
    # Ensure categorical color column is string to avoid double colorbars
    df_plot = df.copy()
    if color_col and color_col in df_plot.columns:
        df_plot[color_col] = df_plot[color_col].astype(str)

    # Sampling for large datasets
    if len(df_plot) > max_scatter_points:
        print(f"Sampling {max_scatter_points} points for scatter plot (original: {len(df_plot)})...")
        df_scatter = df_plot.sample(n=max_scatter_points, random_state=random_seed)
    else:
        df_scatter = df_plot

    # Resolve projection method fallback if missing
    x_cols = [c for c in df.columns if c.startswith("proj_") and c.endswith("_x")]
    requested_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    if requested_col not in df.columns and x_cols:
        fallback_col = x_cols[0]
        if fallback_col.startswith("proj_problem_"):
            method = fallback_col[len("proj_problem_"):-2]
        else:
            method = fallback_col[len("proj_"):-2]

    x_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df.columns else f"proj_{method}_x"
    y_col = f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df.columns else f"proj_{method}_y"

    # Add hover data if available
    hover_cols = ["title", "publication_year", "cited_by_count", "id", "problem", "method", "finding", "interpretation", "doi"]
    hover_cols = [c for c in hover_cols if c in df_scatter.columns]
    
    display_hover = ["title", "publication_year", "cited_by_count"]
    display_hover = [c for c in display_hover if c in df_scatter.columns]

    scatter = px.scatter(
        df_scatter, x=x_col, y=y_col,
        color=color_col, symbol=symbol_col,
        opacity=0.3,
        render_mode="svg",
        color_discrete_sequence=px.colors.qualitative.Set1,
        hover_data=hover_cols
    )

    for tr in scatter.data:
        tr.marker.size = 6
        tr.showlegend = False
        if hover_cols:
            tr.hovertemplate = "<br>".join([f"<b>{c}:</b> %{{customdata[{hover_cols.index(c)}]}}" for c in display_hover]) + "<extra></extra>"
        fig.add_trace(tr)
        
    # 2.5 Add Relevant Papers ("Cities")
    if show_papers:
        import datetime
        from edel.dashboard.utils import parse_authorships
        
        df_papers = df_plot.copy()
        
        if papers_metric == "cited_by_count":
            df_papers = df_papers.sort_values(by="cited_by_count", ascending=False)
            top_papers = df_papers.head(top_papers_n)
        elif papers_metric == "citation_velocity":
            current_year = datetime.datetime.now().year
            years_since = np.maximum(1, current_year - df_papers["publication_year"].fillna(current_year))
            df_papers["velocity"] = df_papers["cited_by_count"].fillna(0) / years_since
            df_papers = df_papers.sort_values(by="velocity", ascending=False)
            top_papers = df_papers.head(top_papers_n)
        elif papers_metric == "citation_normalized_percentile":
            if "citation_normalized_percentile" in df_papers.columns:
                df_papers = df_papers.sort_values(by="citation_normalized_percentile", ascending=False)
            else:
                df_papers = df_papers.sort_values(by="cited_by_count", ascending=False)
            top_papers = df_papers.head(top_papers_n)
        elif papers_metric == "cluster_centroids":
            if color_col and color_col in df_papers.columns:
                clusters = df_papers[color_col].unique()
                k = max(1, top_papers_n // max(1, len(clusters)))
                top_papers = df_papers.sort_values("cited_by_count", ascending=False).groupby(color_col).head(k).head(top_papers_n)
            else:
                top_papers = df_papers.sort_values(by="cited_by_count", ascending=False).head(top_papers_n)
        elif papers_metric == "local_peaks":
            if zi is not None:
                x_min = xi.min()
                y_min = yi.min()
                if len(xi.shape) > 1:
                    dx = xi[0,1] - xi[0,0] if xi.shape[1] > 1 else 1.0
                else:
                    dx = xi[1] - xi[0] if len(xi) > 1 else 1.0
                    
                if len(yi.shape) > 1:
                    dy = yi[1,0] - yi[0,0] if yi.shape[0] > 1 else 1.0
                else:
                    dy = yi[1] - yi[0] if len(yi) > 1 else 1.0
                c_ix = np.clip(np.floor((df_papers[x_col] - (x_min - dx/2)) / dx).astype(int), 0, zi.shape[1]-1)
                c_iy = np.clip(np.floor((df_papers[y_col] - (y_min - dy/2)) / dy).astype(int), 0, zi.shape[0]-1)
                df_papers["peak_height"] = zi[c_iy, c_ix]
                df_papers = df_papers.sort_values(["peak_height", "cited_by_count"], ascending=[False, False])
                top_papers = df_papers.head(top_papers_n)
            else:
                top_papers = df_papers.sort_values(by="cited_by_count", ascending=False).head(top_papers_n)
        else:
            top_papers = df_papers.sort_values(by="cited_by_count", ascending=False).head(top_papers_n)

        labels = []
        for _, row in top_papers.iterrows():
            authors_list = parse_authorships(row.get("authorships", ""))
            first_author = authors_list[0].get("name", "Unknown") if authors_list else "Unknown"
            last_name = first_author.split()[-1] if first_author != "Unknown" else "Unknown"
            year = str(row.get("publication_year", ""))[:4]
            labels.append(f"{last_name} ({year})")
            
        fig.add_trace(go.Scatter(
            x=top_papers[x_col],
            y=top_papers[y_col],
            mode='markers+text',
            marker=dict(size=8, color='lightgray', line=dict(width=1, color='black')),
            text=labels,
            textposition="top center",
            textfont=dict(color="white", size=12, family="Arial Black"),
            name="Relevant Papers",
            showlegend=False,
            hoverinfo='skip',
            cliponaxis=False
        ))

    # Add Region Boundaries and Labels
    if show_regions:
        boundaries = terrain.get("boundaries", {})
        centroids = terrain.get("centroids", {})
        
        for cluster, paths in boundaries.items():
            # Draw boundary paths
            for i, path in enumerate(paths):
                fig.add_trace(
                    go.Scatter(
                        x=path["x"], y=path["y"],
                        mode='lines',
                        line=dict(color="white", width=1.5, dash="dash"),
                        name="Domain Regions",
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
            
            # Add text label at centroid
            if cluster in centroids:
                cx, cy = centroids[cluster]["x"], centroids[cluster]["y"]
                
                # Fetch friendly label if available
                display_name = cluster
                if label_results and color_col == "cluster_domain":
                    domain_info = label_results.get("clusters", {}).get("domain", {})
                    # The ID could be string or int depending on clustering source
                    # We check string first, then int if possible
                    c_info = domain_info.get(cluster)
                    if not c_info:
                        try:
                            c_info = domain_info.get(int(cluster))
                        except ValueError:
                            pass
                    if c_info and "proposed_label" in c_info:
                        display_name = c_info["proposed_label"]

                fig.add_annotation(
                    x=cx, y=cy,
                    text=display_name,
                    showarrow=False,
                    font=dict(size=12, color="white", weight="bold"),
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.3)",
                    borderwidth=1,
                    borderpad=3
                )

    # 3. Add Vector Field Overlay
    if show_flow:
        smoothed_vf = landscape_results.get("vector_field")
        if smoothed_vf:
            # Use the pre-computed flow from Stage 8 (respects config type and boundaries)
            _add_precomputed_flow(fig, smoothed_vf, scale=flow_scale, width=flow_width, arrow_size=arrow_size)
        elif field is not None:
            # Fallback: Compute on the fly (might be slightly misaligned with boundaries)
            _add_flow_to_contour(fig, field, flow_type, scale=flow_scale, width=flow_width, arrow_size=arrow_size)

    _add_manual_legends_2d(fig, df_plot, color_col, symbol_col, label_results)

    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]

    fig.update_layout(
        title=title or f"2D Epistemic Landscape Contour Map: {topic_name}" if topic_name else "2D Epistemic Landscape Contour Map",
        xaxis_title=x_label, yaxis_title=y_label,
        xaxis=dict(range=[x_min, x_max], showgrid=True, zeroline=False),
        yaxis=dict(range=[y_min, y_max], showgrid=True, zeroline=False),
        width=1280, height=800, # 16:10 aspect ratio
        autosize=False,
        legend=dict(
            x=0.98, y=0.98, 
            xanchor="right", yanchor="top", 
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=80, r=100, b=80, t=80) 
    )

    # 4. Add Pre-allocated Trajectory Placeholders (for callbacks)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", 
                             line=dict(color="red", width=3, dash="dot"), 
                             name="Trajectory Line", showlegend=False, hoverinfo="skip", visible=False))
                             
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", 
                             marker=dict(size=14, color="yellow", line=dict(width=3, color="red")), 
                             name="Selected Paper", showlegend=False, hoverinfo="skip", visible=False))

    # Add 3 empty annotations for 2D arrows
    for _ in range(3):
        fig.add_annotation(
            x=0, y=0, ax=0, ay=0, 
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=False, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="red", 
            visible=False
        )

    return fig


# --- Private Helpers ---

def _add_manual_legends_3d(fig, df, color_col, symbol_col, label_results=None):
    colors = px.colors.qualitative.Set1
    symbols = ["circle", "diamond", "square", "cross", "x", "triangle-up"]
    
    if color_col and color_col in df.columns:
        labels = sorted([l for l in df[color_col].unique() if pd.notna(l)], key=str)
        for i, lab in enumerate(labels):
            name = str(lab)
            if name == "-1" or name == "-1.0":
                name = "No cluster"

            if label_results and color_col == "cluster_domain":
                # results["clusters"]["domain"][cluster_id] = {"proposed_label": "..."}
                domain_clusters = label_results.get("clusters", {}).get("domain", {})
                try:
                    cid_str = str(int(float(lab)))
                    # Check both int and string keys as JSON might load as either
                    c_info = domain_clusters.get(cid_str) or domain_clusters.get(int(cid_str))
                    if c_info and "proposed_label" in c_info:
                        name = c_info["proposed_label"]
                except: pass

            legend_title = color_col
            if color_col == "cluster_domain":
                legend_title = "Research Domains"

            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", 
                marker=dict(size=6, color=colors[i % len(colors)]),
                legendgroup="color", legendgrouptitle_text=legend_title, name=name))

    if symbol_col and symbol_col in df.columns:
        labels = df[symbol_col].unique()
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", 
                marker=dict(size=6, color="gray", symbol=symbols[i % len(symbols)]),
                legendgroup="symbol", legendgrouptitle_text=symbol_col, name=str(lab)))

def _add_manual_legends_2d(fig, df, color_col, symbol_col, label_results=None):
    colors = px.colors.qualitative.Set1
    symbols = ["circle", "diamond", "square", "cross", "x", "triangle-up"]
    
    if color_col and color_col in df.columns:
        labels = sorted([l for l in df[color_col].unique() if pd.notna(l)], key=str)
        for i, lab in enumerate(labels):
            name = str(lab)
            if name == "-1" or name == "-1.0":
                name = "No cluster"

            if label_results and color_col == "cluster_domain":
                domain_clusters = label_results.get("clusters", {}).get("domain", {})
                try:
                    cid_str = str(int(float(lab)))
                    c_info = domain_clusters.get(cid_str) or domain_clusters.get(int(cid_str))
                    if c_info and "proposed_label" in c_info:
                        name = c_info["proposed_label"]
                except: pass

            legend_title = color_col
            if color_col == "cluster_domain":
                legend_title = "Research Domains"

            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", 
                marker=dict(size=8, color=colors[i % len(colors)]),
                legendgroup="color", legendgrouptitle_text=legend_title, name=name))

    if symbol_col and symbol_col in df.columns:
        labels = df[symbol_col].unique()
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", 
                marker=dict(size=8, color="gray", symbol=symbols[i % len(symbols)]),
                legendgroup="symbol", legendgrouptitle_text=symbol_col, name=str(lab)))

def _add_precomputed_flow(fig, smoothed_vf, scale=0.20, step=2, width=0.7, arrow_size=0.8):
    """Draw arrows from a pre-computed grid (Stage 8 results)."""
    xi, yi = smoothed_vf["x"], smoothed_vf["y"]
    ui, vi = smoothed_vf["u"], smoothed_vf["v"]

    # Subsample
    xs, ys = xi[::step, ::step].flatten(), yi[::step, ::step].flatten()
    us, vs = ui[::step, ::step].flatten(), vi[::step, ::step].flatten()

    # Filter small/null vectors
    mag = np.sqrt(us**2 + vs**2)
    mask = (mag > 1e-5) & (~np.isnan(mag))
    xs, ys, us, vs = xs[mask], ys[mask], us[mask], vs[mask]

    # Dynamic scaling for annotations
    plot_range = xi.max() - xi.min()
    adjusted_scale = scale * (plot_range / 5.0)

    for x, y, u, v in zip(xs, ys, us, vs):
        # Normalize and scale for consistent arrow lengths if desired, 
        # or use raw scaled vectors for 'speed' visualization
        m_i = np.sqrt(u**2 + v**2)
        if m_i == 0: continue
        dx_s = (u / m_i) * adjusted_scale
        dy_s = (v / m_i) * adjusted_scale

        # CLIP: Ensure the arrow end point is within the landscape boundaries
        end_x, end_y = x + dx_s, y + dy_s
        if end_x < xi.min() or end_x > xi.max() or end_y < yi.min() or end_y > yi.max():
            continue

        fig.add_annotation(
            x=end_x, y=end_y,
            ax=x, ay=y,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,
            arrowsize=arrow_size,
            arrowwidth=width,
            arrowcolor="rgba(0, 0, 0, 0.7)",
            opacity=0.7
        )


def _add_flow_to_contour(fig, field, field_type, scale=0.20, grid_res=40, sigma=0.25, step=2, width=0.7, arrow_size=0.8):
    # Kernel smoothing for beautiful flow
    X = field["cell_px"].values
    Y = field["cell_py"].values
    
    if field_type == "total":
        stages = ["pm", "mf", "fi"]
    elif field_type == "discovery":
        stages = ["mf", "fi"]
    else:
        stages = [field_type]

    DX = np.zeros_like(X)
    DY = np.zeros_like(Y)
    
    for stage in stages:
        col_x, col_y = f"vf_{stage}_x", f"vf_{stage}_y"
        if col_x in field.columns:
            DX += field[col_x].values
            DY += field[col_y].values

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

        # CLIP: Ensure the arrow end point is within the landscape boundaries
        end_x, end_y = x + dx_s, y + dy_s
        if end_x < X.min() or end_x > X.max() or end_y < Y.min() or end_y > Y.max():
            continue

        fig.add_annotation(
            x=end_x, y=end_y,
            ax=x, ay=y,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,
            arrowsize=arrow_size,
            arrowwidth=width,
            arrowcolor="rgba(0, 0, 0, 0.7)",
            opacity=0.7
        )
