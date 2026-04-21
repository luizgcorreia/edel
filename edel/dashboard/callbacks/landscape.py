"""Callbacks for the Interactive Landscape Map panel."""

from pathlib import Path
import numpy as np
import pandas as pd
import dash
from dash import Dash, Input, Output, State, Patch, callback_context, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from edel.experiments.registry import get_experiment
from edel.io.artifact import make_stage_artifact, load_artifact
from edel.viz.landscape import plot_landscape_3d, plot_landscape_contour

_DATASET_CACHE = {}

def register_landscape_callbacks(app: Dash, base_path: Path) -> None:
    
    @app.callback(
        Output("landscape-graph", "figure"),
        [Input("map-experiment-select", "value"),
         Input("map-method-select", "value"),
         Input("map-view-mode", "value"),
         Input("map-layer-toggles", "value")]
    )
    def update_map_figure(experiment_name, method, view_mode, layers):
        """Load artifacts and build the Plotly figure (3D or 2D)."""
        if not experiment_name:
            raise PreventUpdate
            
        try:
            config = get_experiment(experiment_name)
            
            # Load necessary artifacts
            df_art = make_stage_artifact(config, base_path, "clustering", "clustering")
            df = load_artifact(df_art)
            
            land_art = make_stage_artifact(config, base_path, "output", "landscape_results")
            landscape_results = load_artifact(land_art)
            
            label_art = make_stage_artifact(config, base_path, "labeling", "labeled")
            label_results = None
            try:
                label_results = load_artifact(label_art)
            except:
                pass # Labels might not exist yet
                
            # Choose plot function
            if view_mode == "2d":
                fig = plot_landscape_contour(
                    df=df,
                    landscape_results=landscape_results,
                    method=method,
                    color_col="cluster_domain",
                    label_results=label_results,
                    title=f"2D Epistemic Landscape: {experiment_name}"
                )
            else:
                fig = plot_landscape_3d(
                    df=df,
                    landscape_results=landscape_results,
                    method=method,
                    color_col="cluster_domain",
                    label_results=label_results,
                    title=f"3D Epistemic Landscape: {experiment_name}"
                )
            
            if fig is None:
                return go.Figure().update_layout(title="Could not generate landscape plot (missing data).")
            
            # --- Apply Layer Toggles ---
            show_surface = "surface" in layers if layers else True
            show_scatter = "scatter" in layers if layers else True
            show_vectors = "vectors" in layers if layers else False
            show_clusters = "clusters" in layers if layers else True

            for trace in fig.data:
                # Identify traces by name or type
                t_name = trace.name.lower() if trace.name else ""
                t_type = trace.type.lower()
                
                if "surface" in t_type or "contour" in t_type or "terrain" in t_name:
                    trace.visible = show_surface
                elif "scatter" in t_type or "markers" in trace.mode:
                    # If it's a manual legend trace (no data points), control by show_clusters
                    if trace.showlegend and not trace.x: 
                        trace.visible = show_clusters
                    else:
                        trace.visible = show_scatter
                elif "vector" in t_name or "flow" in t_name:
                    trace.visible = show_vectors

            # Handle annotations (vector field arrows in 2D are annotations)
            if view_mode == "2d":
                for ann in fig.layout.annotations:
                    ann.visible = show_vectors

            # Add customdata to scatter traces so we can extract paper IDs on click
            # (Already handled in plot_landscape_3d/contour usually, but let's ensure)
            
            return fig
            
        except Exception as e:
            print(f"Error rendering landscape: {e}")
            return go.Figure().update_layout(title=f"Error loading map data: {str(e)}")

    # (Removed redundant toggle_layers callback as it's now integrated into update_map_figure)

    # --- Server-Side Search Callback ---
    @app.callback(
        Output("map-paper-search", "options"),
        [Input("map-paper-search", "search_value"),
         Input("map-experiment-select", "value")]
    )
    def update_search_options(search_value, experiment_name):
        if not search_value or not experiment_name or len(search_value) < 2:
            raise PreventUpdate

        # 1. Lazy load and cache the dataset
        if experiment_name not in _DATASET_CACHE:
            config = get_experiment(experiment_name)
            df_art = make_stage_artifact(config, base_path, "clustering", "clustering")
            try:
                _DATASET_CACHE[experiment_name] = load_artifact(df_art)
            except Exception as e:
                print(f"Error loading artifact for search: {e}")
                return []
        
        df = _DATASET_CACHE.get(experiment_name)
        if df is None or df.empty:
            return []
        
        # 2. Filter efficiently (case-insensitive substring match on title or exact match on ID)
        search_value = search_value.lower()
        
        # Handle cases where title might be missing
        title_match = df["title"].str.lower().str.contains(search_value, na=False)
        id_match = df["id"] == search_value
        mask = title_match | id_match
        
        matches = df[mask].head(50) # Return max 50 results
        
        options = [{"label": row.get("title", "Unknown"), "value": row["id"]} for _, row in matches.iterrows()]
        return options

    # --- Click & Selection Callback ---
    @app.callback(
        [Output("map-selected-paper-info", "children"),
         Output("landscape-graph", "figure", allow_duplicate=True),
         Output("map-paper-search", "value")],
        [Input("landscape-graph", "clickData"),
         Input("map-paper-search", "value")],
        [State("map-experiment-select", "value"),
         State("map-method-select", "value"),
         State("map-view-mode", "value")],
        prevent_initial_call=True
    )
    def display_click_data(clickData, search_val, experiment_name, method, view_mode):
        """Display info when a paper is clicked or searched and draw its trajectory."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        paper_id = None
        clicked_x = None
        clicked_y = None
        clicked_z = None
        
        if triggered_id == "landscape-graph" and clickData:
            try:
                point = clickData["points"][0]
                if "customdata" in point:
                    paper_id = point["customdata"][3] # URL/ID is at index 3
                    search_val = paper_id # Sync dropdown
                    clicked_x = point.get("x")
                    clicked_y = point.get("y")
                    clicked_z = point.get("z")
            except Exception as e:
                print(f"Error extracting click data: {e}")
                
        elif triggered_id == "map-paper-search" and search_val:
            paper_id = search_val
            
        if not paper_id:
            return "Select a paper to view its trajectory.", dash.no_update, None
            
        try:
            # We need the full dataframe for trajectory coordinates
            if experiment_name not in _DATASET_CACHE:
                config = get_experiment(experiment_name)
                df_art = make_stage_artifact(config, base_path, "clustering", "clustering")
                _DATASET_CACHE[experiment_name] = load_artifact(df_art)
            
            df = _DATASET_CACHE[experiment_name]
            matches = df[df["id"] == paper_id]
            if matches.empty:
                return "Paper not found in dataset.", dash.no_update, search_val
                
            row = matches.iloc[0]
            
            title = row.get("title", "Unknown Title")
            year = row.get("publication_year", "Unknown Year")
            cits = row.get("cited_by_count", "0")
            url = row.get("id", "#")
            prob = row.get("problem", "")
            meth = row.get("method", "")
            find = row.get("finding", "")
            inter = row.get("interpretation", "")
            doi = row.get("doi", "")
            
            # If not clicked from map, fallback to dataset coordinates
            if clicked_x is None:
                clicked_x = row.get(f"proj_problem_{method}_x", None)
                clicked_y = row.get(f"proj_problem_{method}_y", None)
            
            # --- 1. Info Panel ---
            def fmt_coord(v):
                if v is None: return "N/A"
                try: return f"{float(v):.2f}"
                except: return "N/A"

            z_coord = clicked_z
            if z_coord is None:
                try: z_coord = np.log10(float(cits) + 1)
                except: z_coord = "N/A"

            info = [
                html.H5(title, className="mb-1"),
                html.P(f"({year}) • {cits} citations", className="text-muted small"),
                html.Hr(),
                html.Div([
                    html.B("Problem: "), html.Span(prob), html.Br(),
                    html.B("Method: "), html.Span(meth), html.Br(),
                    html.B("Finding: "), html.Span(find), html.Br(),
                    html.B("Interpretation: "), html.Span(inter),
                ], className="small mb-3"),
                html.Div([
                    html.A("View on OpenAlex", href=url, target="_blank", className="btn btn-sm btn-outline-primary me-2"),
                    html.A("DOI", href=f"https://doi.org/{doi}", target="_blank", className="btn btn-sm btn-outline-secondary") if doi else None
                ]),
                html.Hr(),
                html.P(f"Coordinates: ({fmt_coord(clicked_x)}, {fmt_coord(clicked_y)}, {fmt_coord(z_coord)})", className="text-muted tiny")
            ]
            
            # --- 2. Trajectory Figure Update ---
            aspects = ["problem", "method", "finding", "interpretation"]
            patched_fig = Patch()
            
            xs, ys, zs = [], [], []
            for i, asp in enumerate(aspects):
                if i == 0 and clicked_x is not None and clicked_y is not None:
                    # Use exact clicked point for the initial 'P' node
                    x, y = clicked_x, clicked_y
                else:
                    x = row[f"proj_{asp}_{method}_x"]
                    y = row[f"proj_{asp}_{method}_y"]
                
                # Ensure Z coordinate is consistent
                try: z = float(z_coord)
                except: z = np.log10(float(cits) + 1)
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
            
            if view_mode == "3d":
                # Trace -1: Highlight, Trace -2: Line
                patched_fig["data"][-1]["x"] = [xs[0]]
                patched_fig["data"][-1]["y"] = [ys[0]]
                patched_fig["data"][-1]["z"] = [zs[0]]
                patched_fig["data"][-1]["visible"] = True
                
                patched_fig["data"][-2]["x"] = xs
                patched_fig["data"][-2]["y"] = ys
                patched_fig["data"][-2]["z"] = zs
                patched_fig["data"][-2]["visible"] = True
            else:
                # 2D Mode
                # Trace -1: Highlight, Trace -2: Line
                patched_fig["data"][-1]["x"] = [xs[0]]
                patched_fig["data"][-1]["y"] = [ys[0]]
                patched_fig["data"][-1]["visible"] = True
                
                patched_fig["data"][-2]["x"] = xs
                patched_fig["data"][-2]["y"] = ys
                patched_fig["data"][-2]["visible"] = True
                
                # Update Annotations for Arrows (Last 3 annotations)
                for i in range(3):
                    idx = -(3 - i) # -3, -2, -1
                    patched_fig["layout"]["annotations"][idx]["x"] = xs[i+1]
                    patched_fig["layout"]["annotations"][idx]["y"] = ys[i+1]
                    patched_fig["layout"]["annotations"][idx]["ax"] = xs[i]
                    patched_fig["layout"]["annotations"][idx]["ay"] = ys[i]
                    patched_fig["layout"]["annotations"][idx]["showarrow"] = True
                    patched_fig["layout"]["annotations"][idx]["visible"] = True
            
            return html.Div(info), patched_fig, search_val
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error parsing paper data: {e}", dash.no_update, search_val
