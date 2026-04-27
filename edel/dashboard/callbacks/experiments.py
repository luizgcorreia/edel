"""Callbacks for the Experiment Runner and Stage Debugger panels."""

from pathlib import Path
from dash import Dash, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import pandas as pd
from dash import dash_table

from edel.experiments.registry import get_experiment
from edel.experiments.snippets import get_snippets, save_snippet, STAGE_KEYS, STAGE_LIST
from edel.dashboard.worker import submit_job, list_jobs
from edel.io.artifact import make_stage_artifact, load_artifact, save_artifact, CANONICAL_ARTIFACT_NAMES
from edel.dashboard.utils import df_to_dash_columns, df_to_dash_records
import edel.pipeline as pipeline
import edel.viz as viz
import itertools
import copy
import json
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def register_experiment_callbacks(app: Dash, base_path: Path) -> None:
    
    # --- Helper: Generate IDs for callbacks ---
    AXIS_IDS = [f"sweep-{s.lower().replace(' ', '-')}" for s in STAGE_LIST]

    # ---------------------------------------------------------------------------
    # Experiment Runner Callbacks
    # ---------------------------------------------------------------------------
    
    # Combine all inputs into a single flat list for the decorator
    SUMMARY_INPUTS = [Input("sweep-base-config", "value")] + [Input(id_name, "value") for id_name in AXIS_IDS]

    @app.callback(
        Output("sweep-summary", "children"),
        SUMMARY_INPUTS
    )
    def update_sweep_summary(*args):
        """Update the text summarizing how many jobs will be created."""
        if not args or not args[0]:
            return "Select a base config."
        
        base_name = args[0]
        stage_values = args[1:]
        
        n_jobs = 1
        for val in stage_values:
            if val:
                n_jobs *= len(val)
        
        return html.Div([
            html.Span(f"→ {n_jobs} job(s) will be created.", className="fw-bold"),
            html.Br(),
            html.Small("Cartesian product of all selected snippets.", className="text-muted")
        ])

    # Define inputs and states separately for the standard Dash signature
    QUEUE_INPUTS = [
        Input("btn-submit-jobs", "n_clicks"),
        Input("job-queue-interval", "n_intervals"),
        Input("btn-refresh-jobs", "n_clicks")
    ]
    QUEUE_STATES = [
        State("sweep-base-config", "value")
    ] + [State(id_name, "value") for id_name in AXIS_IDS]

    @app.callback(
        Output("job-queue-table", "data"),
        QUEUE_INPUTS,
        QUEUE_STATES,
        prevent_initial_call=False
    )
    def update_job_queue(*args):
        """Submit jobs if clicked, then return current queue status."""
        # Unpack arguments
        submit_clicks = args[0]
        n_intervals = args[1]
        refresh_clicks = args[2]
        base_name = args[3]
        stage_values = args[4:]
        
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        if triggered_id == "btn-submit-jobs" and base_name:
            try:
                base_config = get_experiment(base_name)
                
                # Build the axes list for cartesian product
                axes_snippets = []
                axes_keys = []
                
                for i, stage_name in enumerate(STAGE_LIST):
                    selected_names = stage_values[i]
                    if selected_names:
                        stage_snippets = get_snippets(stage_name)
                        axes_snippets.append([stage_snippets[name] for name in selected_names])
                        axes_keys.append(STAGE_KEYS[stage_name])
                
                if not axes_snippets:
                    # Just submit the base config as one job
                    submit_job(base_config, base_path)
                else:
                    # Generate combinations
                    for combo in itertools.product(*axes_snippets):
                        config = copy.deepcopy(base_config)
                        # Merge each snippet in the combination
                        for key, snippet in zip(axes_keys, combo):
                            if key in config and isinstance(config[key], dict) and isinstance(snippet, dict):
                                config[key].update(snippet)
                            else:
                                config[key] = snippet
                        submit_job(config, base_path)
                        
            except Exception as e:
                print(f"Error submitting jobs: {e}")
                
        # Always fetch latest job list
        jobs = list_jobs(base_path)
        table_data = []
        for j in jobs:
            table_data.append({
                "job_id": j.get("job_id", ""),
                "experiment_id": j.get("experiment_id", "Unknown"),
                "status": j.get("status", "unknown"),
                "submitted_at": j.get("submitted_at", "").split("T")[0] + " " + j.get("submitted_at", "").split("T")[1][:8] if "T" in j.get("submitted_at", "") else j.get("submitted_at", "")
            })
        return table_data

    # --- Snippet Management Callbacks ---

    @app.callback(
        Output("snippet-modal", "is_open"),
        [Input("btn-open-snippet-modal", "n_clicks"),
         Input("btn-save-snippet", "n_clicks")],
        [State("snippet-modal", "is_open"),
         State("snippet-stage-select", "value"),
         State("snippet-name-input", "value"),
         State("snippet-config-input", "value")],
        prevent_initial_call=True
    )
    def toggle_snippet_modal(n_open, n_save, is_open, stage, name, config_json):
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == "btn-save-snippet" and stage and name and config_json:
            try:
                config = json.loads(config_json)
                save_snippet(stage, name, config)
            except Exception as e:
                print(f"Error saving snippet: {e}")
            return False # Close modal on save
            
        return not is_open

    # ---------------------------------------------------------------------------
    # Stage Debugger Callbacks
    # ---------------------------------------------------------------------------
    
    def capture_matplotlib_plot(func, *args, **kwargs):
        """Helper to capture a matplotlib plot as a Dash html.Img."""
        # Use a non-interactive backend for thread safety in web apps if possible, 
        # but here we just ensure we clear the figure.
        plt.figure(figsize=(10, 6))
        
        # Patch plt.show temporarily
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            func(*args, **kwargs)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode('utf-8')
            return html.Img(src=f"data:image/png;base64,{encoded}", className="img-fluid mb-4 shadow-sm rounded")
        except Exception as e:
            plt.close()
            return html.Div(f"Error generating plot: {str(e)}", className="text-warning small")
        finally:
            plt.show = original_show

    @app.callback(
        [Output("debug-data-container", "children"),
         Output("debug-artifact-info", "children")],
        [Input("btn-load-artifact", "n_clicks"),
         Input("btn-run-stage", "n_clicks")],
        [State("debug-experiment-select", "value"),
         State("debug-stage-select", "value")],
        prevent_initial_call=True
    )
    def handle_debug_action(load_clicks, run_clicks, experiment_name, stage_name):
        """Load or Run an intermediate stage for debugging."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if not experiment_name or not stage_name:
            raise PreventUpdate
            
        try:
            config = get_experiment(experiment_name)
            
            if triggered_id == "btn-run-stage":
                # --- RUN STAGE LOGIC (Mirroring exploration.ipynb) ---
                data = None
                field = None
                
                # 1. Load dependencies
                if stage_name == "data_collection":
                    data = pipeline.run_data_stage(config)
                elif stage_name == "structured_abstracts":
                    prev_art = make_stage_artifact(config, base_path, "data_collection", "dataset")
                    data = pipeline.run_structuring_stage(load_artifact(prev_art), config)
                elif stage_name == "embeddings":
                    prev_art = make_stage_artifact(config, base_path, "structured_abstracts", "sa")
                    prev_data = load_artifact(prev_art)
                    if isinstance(prev_data, tuple):
                        prev_data = prev_data[0]
                    data = pipeline.run_embedding_stage(prev_data, config)
                elif stage_name == "dimensionality_reduction":
                    prev_art = make_stage_artifact(config, base_path, "embeddings", "embeddings")
                    data = pipeline.run_projection_stage(load_artifact(prev_art), config)
                elif stage_name == "vector_field":
                    prev_art = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
                    field = pipeline.run_vector_field_stage(load_artifact(prev_art), config)
                    data = field # For generic display
                elif stage_name == "clustering":
                    art_df = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
                    art_field = make_stage_artifact(config, base_path, "vector_field", "field")
                    data, field = pipeline.run_clustering_stage(load_artifact(art_df), load_artifact(art_field), config)
                elif stage_name == "labeling":
                    art_df = make_stage_artifact(config, base_path, "clustering", "clustering")
                    art_field = make_stage_artifact(config, base_path, "clustering", "field_clustering")
                    from edel.pipeline.labeling import get_llm_client
                    llm_client = get_llm_client(config.get("labeling", {}))
                    data = pipeline.run_labeling_stage(load_artifact(art_df), load_artifact(art_field), config, llm_client)
                elif stage_name == "output":
                    art_df = make_stage_artifact(config, base_path, "clustering", "clustering")
                    art_field = make_stage_artifact(config, base_path, "vector_field", "field")
                    data = pipeline.run_landscape_stage(load_artifact(art_df), load_artifact(art_field), config)
                
                # 2. Save artifacts
                art_names = CANONICAL_ARTIFACT_NAMES.get(stage_name, [])
                saved_info = ""
                if stage_name == "clustering" and isinstance(data, pd.DataFrame) and isinstance(field, pd.DataFrame):
                    p1 = save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[0]), data)
                    p2 = save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[1]), field)
                    saved_info = str(p1.parent)
                elif data is not None:
                    p = save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[0]), data)
                    saved_info = str(p)
                
                info_prefix = f"Ran and Saved: {saved_info}"
            else:
                # --- LOAD ARTIFACT LOGIC ---
                art_names = CANONICAL_ARTIFACT_NAMES.get(stage_name, [])
                art_name = art_names[0]
                artifact = make_stage_artifact(config, base_path, stage_name, art_name)
                data = load_artifact(artifact)
                # Robustness: some stages might have saved tuples (df, report)
                if isinstance(data, tuple) and len(data) > 0 and isinstance(data[0], pd.DataFrame):
                    data = data[0]
                field = None
                
                # Find which one exists
                p = artifact.parquet_path if artifact.parquet_path.exists() else artifact.pkl_path
                info_prefix = f"Loaded: {p}"

            # --- VISUALIZATION LOGIC ---
            viz_components = []
            
            if stage_name == "data_collection":
                viz_components.append(capture_matplotlib_plot(viz.plot_publication_year_dist, data))
                viz_components.append(capture_matplotlib_plot(viz.plot_citation_dist, data))
                viz_components.append(capture_matplotlib_plot(viz.plot_abstract_length_dist, data))
            
            elif stage_name == "dimensionality_reduction":
                method = config.get("dimensionality_reduction", {}).get("method", "umap")
                viz_components.append(capture_matplotlib_plot(viz.plot_projection_2d, data, method=method))
                viz_components.append(capture_matplotlib_plot(viz.plot_paper_style_pca, data))
                
                # Transition space plot
                n_dims = config.get("embedding", {}).get("n_dimensions", 1536)
                viz_components.append(capture_matplotlib_plot(viz.plot_epistemic_transition_space, data, quantile=0.95, dimensions=n_dims))
            
            elif stage_name == "vector_field":
                # Need DF and field for some vector field plots
                art_df = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
                df_proj = load_artifact(art_df)
                viz_components.append(capture_matplotlib_plot(viz.plot_vector_field, data))
                viz_components.append(capture_matplotlib_plot(viz.plot_field_magnitude, data))
                viz_components.append(capture_matplotlib_plot(viz.plot_transition_signatures, df_proj))
                viz_components.append(capture_matplotlib_plot(viz.plot_movement_magnitudes, df_proj))
            
            elif stage_name == "clustering":
                method = config.get("dimensionality_reduction", {}).get("method", "umap")
                viz_components.append(capture_matplotlib_plot(viz.plot_clusters_on_landscape, data, method=method, cluster_key="domain"))
                # If we have field clustering
                if field is not None or triggered_id == "btn-load-artifact":
                    try:
                        f_clust = load_artifact(make_stage_artifact(config, base_path, stage_name, "field_clustering"))
                        viz_components.append(capture_matplotlib_plot(viz.plot_field_clusters, f_clust, cluster_key="field"))
                    except: pass
                viz_components.append(capture_matplotlib_plot(viz.plot_cluster_trajectories, data, method=method, cluster_key="style"))
            
            elif stage_name == "labeling":
                summaries = viz.print_cluster_summaries(data, cluster_key="domain")
                viz_components.append(html.Pre(summaries, className="p-3 bg-light rounded small", style={"whiteSpace": "pre-wrap"}))
            
            elif stage_name == "output":
                method = config.get("dimensionality_reduction", {}).get("method", "umap")
                provider_cfg = config.get("data", {}).get("provider", {})
                topic_name = provider_cfg.get("topic_name") or provider_cfg.get("topic_id", "Unknown")
                
                # Need clustering DF and labeling results for the Epistemic Map
                try:
                    art_clust = make_stage_artifact(config, base_path, "clustering", "clustering")
                    df_clust = load_artifact(art_clust)
                    
                    art_labeled = make_stage_artifact(config, base_path, "labeling", "labeled")
                    label_results = load_artifact(art_labeled)
                    
                    viz_components.append(capture_matplotlib_plot(
                        viz.plot_epistemic_map, 
                        df_clust, 
                        label_results, 
                        method=method, 
                        topic_name=topic_name
                    ))
                except Exception as e:
                    viz_components.append(html.Div(f"Could not load data for Epistemic Map: {e}", className="text-warning small"))

            # --- RENDER TABLE/SUMMARY ---
            if isinstance(data, pd.DataFrame):
                info = f"{info_prefix}: DataFrame ({len(data)} rows × {len(data.columns)} cols)"
                table = dash_table.DataTable(
                    columns=df_to_dash_columns(data),
                    data=df_to_dash_records(data, max_rows=50),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                )
                return html.Div([
                    html.Div(viz_components),
                    html.Hr(),
                    html.P("Showing first 50 rows:", className="text-muted small"),
                    table
                ]), info
                
            elif isinstance(data, dict):
                info = f"{info_prefix}: Dictionary ({len(data)} keys)"
                safe_dict = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v for k, v in data.items()}
                return html.Div([
                    html.Div(viz_components),
                    html.Hr(),
                    html.Pre(json.dumps(safe_dict, indent=2, default=str), style={"maxHeight": "500px", "overflowY": "auto"})
                ]), info
                
            else:
                info = f"{info_prefix}: {type(data).__name__}"
                return html.Div([
                    html.Div(viz_components),
                    html.Hr(),
                    html.Pre(str(data))
                ]), info
                
        except Exception as e:
            return html.Div(f"Error loading artifact: {str(e)}", className="text-danger"), "Error"
