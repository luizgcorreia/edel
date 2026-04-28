"""Callbacks for the Interactive Landscape Map panel."""

from pathlib import Path
import numpy as np
import pandas as pd
import dash
from dash import Dash, Input, Output, State, Patch, callback_context, html, dcc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.io as pio
import io

from edel.experiments.registry import get_experiment
from edel.io.artifact import make_stage_artifact, load_artifact
from edel.viz.landscape import plot_landscape_3d, plot_landscape_contour
from edel.dashboard.utils import parse_authorships

_DATASET_CACHE = {}

def register_landscape_callbacks(app: Dash, base_path: Path) -> None:
    
    @app.callback(
        [Output("landscape-graph-3d", "figure"),
         Output("landscape-graph-2d", "figure")],
        [Input("map-experiment-select", "value"),
         Input("map-method-select", "value"),
         Input("map-layer-toggles", "value"),
         Input("artifact-update-store", "data")]
    )
    def update_map_figures(experiment_name, method, layers, update_signal):
        """Load artifacts and build BOTH Plotly figures (3D and 2D)."""
        if not experiment_name:
            raise PreventUpdate
            
        # Clear cache if triggered by store
        ctx = callback_context
        if ctx.triggered:
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if triggered_id == "artifact-update-store":
                if experiment_name in _DATASET_CACHE:
                    del _DATASET_CACHE[experiment_name]
        
        try:
            config = get_experiment(experiment_name)
            df_art = make_stage_artifact(config, base_path, "clustering", "clustering")
            df = load_artifact(df_art)
            land_art = make_stage_artifact(config, base_path, "output", "landscape_results")
            landscape_results = load_artifact(land_art)
            
            label_art = make_stage_artifact(config, base_path, "labeling", "labeled")
            label_results = None
            try: label_results = load_artifact(label_art)
            except: pass

            # Generate 3D
            fig_3d = plot_landscape_3d(
                df=df, landscape_results=landscape_results, method=method,
                color_col="cluster_domain", label_results=label_results,
                title=f"3D Epistemic Landscape: {experiment_name}"
            )
            
            # Generate 2D
            fig_2d = plot_landscape_contour(
                df=df, landscape_results=landscape_results, method=method,
                color_col="cluster_domain", label_results=label_results,
                title=f"2D Epistemic Landscape: {experiment_name}"
            )

            def apply_layers(fig, is_2d=False):
                if fig is None: return go.Figure()
                show_surface = "surface" in layers if layers else True
                show_scatter = "scatter" in layers if layers else True
                show_vectors = "vectors" in layers if layers else False
                show_clusters = "clusters" in layers if layers else True

                for trace in fig.data:
                    t_name = trace.name.lower() if trace.name else ""
                    t_type = trace.type.lower()
                    if "surface" in t_type or "contour" in t_type or "terrain" in t_name:
                        trace.visible = show_surface
                    elif "scatter" in t_type or "markers" in trace.mode:
                        if trace.showlegend and not (hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0): 
                            trace.visible = show_clusters
                        else:
                            trace.visible = show_scatter
                    elif "vector" in t_name or "flow" in t_name:
                        trace.visible = show_vectors

                if is_2d:
                    for ann in fig.layout.annotations:
                        ann.visible = show_vectors
                return fig

            return apply_layers(fig_3d), apply_layers(fig_2d, is_2d=True)
            
        except Exception as e:
            err_fig = go.Figure().update_layout(title=f"Error: {str(e)}")
            return err_fig, err_fig

    @app.callback(
        [Output("map-container-3d", "style"),
         Output("map-container-2d", "style")],
        [Input("map-view-mode", "value")]
    )
    def toggle_map_visibility(view_mode):
        if view_mode == "2d":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # (Removed redundant toggle_layers callback as it's now integrated into update_map_figure)

    # --- Server-Side Search Callback ---
    @app.callback(
        Output("map-paper-search", "options"),
        [Input("map-paper-search", "search_value"),
         Input("map-experiment-select", "value"),
         Input("map-paper-search", "value")]
    )
    def update_search_options(search_value, experiment_name, current_value):
        if not experiment_name:
            raise PreventUpdate
            
        # 1. Initialize options list
        options = []
        
        # 2. Lazy load and cache the dataset
        if experiment_name not in _DATASET_CACHE:
            config = get_experiment(experiment_name)
            df_art = make_stage_artifact(config, base_path, "clustering", "clustering")
            try:
                _DATASET_CACHE[experiment_name] = load_artifact(df_art)
            except Exception:
                return []
        
        df = _DATASET_CACHE.get(experiment_name)
        if df is None or df.empty:
            return []
            
        # 3. Add matches based on search_value if present
        if search_value and len(search_value) >= 2:
            search_value_low = search_value.lower()
            title_match = df["title"].str.lower().str.contains(search_value_low, na=False)
            id_match = df["id"] == search_value_low
            mask = title_match | id_match
            matches = df[mask].head(50)
            options = [{"label": row.get("title", "Unknown"), "value": row["id"]} for _, row in matches.iterrows()]
        
        # 4. Ensure current selection is in options so the label shows up correctly
        if current_value and current_value not in [o["value"] for o in options]:
            selected_row = df[df["id"] == current_value]
            if not selected_row.empty:
                options.insert(0, {"label": selected_row.iloc[0].get("title", "Selected Paper"), "value": current_value})
                
        return options

    # --- Click & Selection Callback ---
    @app.callback(
        [Output("map-selected-paper-info", "children"),
         Output("landscape-graph-3d", "figure", allow_duplicate=True),
         Output("landscape-graph-2d", "figure", allow_duplicate=True),
         Output("map-paper-search", "value")],
        [Input("landscape-graph-3d", "clickData"),
         Input("landscape-graph-2d", "clickData"),
         Input("map-paper-search", "value")],
        [State("map-experiment-select", "value"),
         State("map-method-select", "value"),
         State("map-view-mode", "value")],
        prevent_initial_call=True
    )
    def display_click_data(click3d, click2d, search_val, experiment_name, method, view_mode):
        """Display info when a paper is clicked or searched and draw its trajectory."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        paper_id = None
        clicked_x = None
        clicked_y = None
        clicked_z = None
        
        clickData = click3d if triggered_id == "landscape-graph-3d" else click2d
        
        if (triggered_id == "landscape-graph-3d" or triggered_id == "landscape-graph-2d") and clickData:
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
            return "Select a paper to view its trajectory.", dash.no_update, dash.no_update, None
            
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

            # Parse authorships
            authorships = row.get("authorships", [])
            parsed_authors = parse_authorships(authorships)
            author_elements = []
            if parsed_authors:
                author_elements.append(html.Div([
                    html.B("Authors: ", style={"fontSize": "0.875rem"}),
                    html.Ul([
                        html.Li([
                            html.A(auth["name"], href=auth["id"], target="_blank", className="text-decoration-none"),
                            html.A(
                                html.Img(src="https://info.orcid.org/wp-content/uploads/2021/12/orcid_16x16.gif", style={"marginLeft": "5px", "verticalAlign": "middle"}),
                                href=auth["orcid"] if str(auth["orcid"]).startswith("http") else f"https://orcid.org/{auth['orcid']}", 
                                target="_blank",
                                className="ms-1"
                            ) if auth.get("orcid") else None,
                            html.Span(f" ({auth['position']}{', corresponding' if auth['is_corresponding'] else ''})", 
                                     className="text-muted", style={"fontSize": "0.75rem", "lineHeight": "1.1"}),
                            html.Br(),
                            html.Small(", ".join([inst["name"] for inst in auth["institutions"]]), 
                                      className="text-muted", style={"fontSize": "0.75rem", "lineHeight": "1.1"})
                        ], className="mb-1") for auth in parsed_authors
                    ], className="ps-3 mb-2", style={"listStyleType": "none", "paddingLeft": "0"})
                ], className="mb-3"))

            info = [
                html.H5(title, className="mb-1"),
                html.P(f"({year}) • {cits} citations", className="text-muted small"),
                html.Hr(),
                html.Div([
                    html.B("Problem: "), html.Span(prob), html.Br(),
                    html.B("Method: "), html.Span(meth), html.Br(),
                    html.B("Finding: "), html.Span(find), html.Br(),
                    html.B("Interpretation: "), html.Span(inter),
                ], style={"fontSize": "0.875rem"}, className="mb-3"),
                *author_elements,
                html.Div([
                    html.A("View on OpenAlex", href=url, target="_blank", className="btn btn-sm btn-outline-primary me-2"),
                    html.A("DOI", href=f"https://doi.org/{doi}", target="_blank", className="btn btn-sm btn-outline-secondary") if doi else None
                ]),
                html.Hr(),
                html.P(f"Coordinates: ({fmt_coord(clicked_x)}, {fmt_coord(clicked_y)}, {fmt_coord(z_coord)})", 
                      className="text-muted", style={"fontSize": "0.75rem"})
            ]
            
            # --- 2. Trajectory Figure Update ---
            aspects = ["problem", "method", "finding", "interpretation"]
            patched_3d = Patch()
            patched_2d = Patch()
            
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
            
            # Update 3D Patch
            patched_3d["data"][-1]["x"] = [xs[0]]
            patched_3d["data"][-1]["y"] = [ys[0]]
            patched_3d["data"][-1]["z"] = [zs[0]]
            patched_3d["data"][-1]["visible"] = True
            patched_3d["data"][-2]["x"] = xs
            patched_3d["data"][-2]["y"] = ys
            patched_3d["data"][-2]["z"] = zs
            patched_3d["data"][-2]["visible"] = True
            
            # Update 2D Patch
            patched_2d["data"][-1]["x"] = [xs[0]]
            patched_2d["data"][-1]["y"] = [ys[0]]
            patched_2d["data"][-1]["visible"] = True
            patched_2d["data"][-2]["x"] = xs
            patched_2d["data"][-2]["y"] = ys
            patched_2d["data"][-2]["visible"] = True
            
            # Update Annotations for Arrows in 2D
            for i in range(3):
                idx = -(3 - i)
                patched_2d["layout"]["annotations"][idx]["x"] = xs[i+1]
                patched_2d["layout"]["annotations"][idx]["y"] = ys[i+1]
                patched_2d["layout"]["annotations"][idx]["ax"] = xs[i]
                patched_2d["layout"]["annotations"][idx]["ay"] = ys[i]
                patched_2d["layout"]["annotations"][idx]["showarrow"] = True
                patched_2d["layout"]["annotations"][idx]["visible"] = True
            
            return html.Div(info), patched_3d, patched_2d, search_val
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error parsing paper data: {e}", dash.no_update, dash.no_update, search_val

    # --- Download Callback ---
    @app.callback(
        Output("download-map-file", "data"),
        [Input("btn-download-png", "n_clicks"),
         Input("btn-download-html", "n_clicks")],
        [State("map-view-mode", "value"),
         State("landscape-graph-3d", "figure"),
         State("landscape-graph-2d", "figure"),
         State("landscape-graph-3d", "relayoutData"),
         State("landscape-graph-2d", "relayoutData"),
         State("map-experiment-select", "value")],
        prevent_initial_call=True
    )
    def download_map(png_clicks, html_clicks, view_mode, fig_3d, fig_2d, relayout_3d, relayout_2d, experiment_name):
        ctx = callback_context
        if not ctx.triggered or not experiment_name:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Select target figure and relayout
        fig_dict = fig_3d if view_mode == "3d" else fig_2d
        relayout = relayout_3d if view_mode == "3d" else relayout_2d
        
        if not fig_dict:
            raise PreventUpdate
            
        # Create a real Figure object from the dict
        fig = go.Figure(fig_dict)
        
        # Apply current zoom/camera from relayoutData
        if relayout:
            for k, v in relayout.items():
                if k == "autosize": continue
                # Handle nested keys like 'scene.camera' or 'xaxis.range'
                keys = k.split('.')
                if len(keys) == 1:
                    fig.update_layout({k: v})
                elif len(keys) == 2:
                    fig.update_layout({keys[0]: {keys[1]: v}})
                elif len(keys) == 3:
                    fig.update_layout({keys[0]: {keys[1]: {keys[2]: v}}})

        filename_base = f"edel_{experiment_name}_{view_mode}"
        
        if triggered_id == "btn-download-html":
            return dcc.send_string(fig.to_html(), f"{filename_base}.html")
        else:
            # PNG Download
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            return dcc.send_bytes(img_bytes, f"{filename_base}.png")
