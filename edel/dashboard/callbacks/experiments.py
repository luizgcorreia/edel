"""Callbacks for the Experiment Runner and Stage Debugger panels."""

from pathlib import Path
from dash import Dash, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import pandas as pd
from dash import dash_table

from edel.experiments.registry import get_experiment
from edel.dashboard.worker import submit_sweep, list_jobs
from edel.io.artifact import make_stage_artifact, load_artifact, CANONICAL_ARTIFACT_NAMES
from edel.dashboard.utils import df_to_dash_columns, df_to_dash_records

def register_experiment_callbacks(app: Dash, base_path: Path) -> None:
    
    # ---------------------------------------------------------------------------
    # Experiment Runner Callbacks
    # ---------------------------------------------------------------------------
    
    @app.callback(
        Output("sweep-summary", "children"),
        [Input("sweep-base-config", "value"),
         Input("sweep-embedding", "value"),
         Input("sweep-projection", "value")]
    )
    def update_sweep_summary(base_name, embeddings, projections):
        """Update the text summarizing how many jobs will be created."""
        if not base_name:
            return "Select a base config."
        
        n_emb = len(embeddings) if embeddings else 1
        n_proj = len(projections) if projections else 1
        n_jobs = n_emb * n_proj
        
        return f"→ {n_jobs} job(s) will be created."

    @app.callback(
        Output("job-queue-table", "data"),
        [Input("btn-submit-jobs", "n_clicks"),
         Input("job-queue-interval", "n_intervals"),
         Input("btn-refresh-jobs", "n_clicks")],
        [State("sweep-base-config", "value"),
         State("sweep-embedding", "value"),
         State("sweep-projection", "value")],
        prevent_initial_call=False
    )
    def update_job_queue(submit_clicks, n_intervals, refresh_clicks, base_name, embeddings, projections):
        """Submit jobs if clicked, then return current queue status."""
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        if triggered_id == "btn-submit-jobs" and base_name:
            try:
                base_config = get_experiment(base_name)
                sweep_axes = {}
                if embeddings:
                    sweep_axes["embedding.model"] = embeddings
                if projections:
                    sweep_axes["dimensionality_reduction.method"] = projections
                
                submit_sweep(base_config, sweep_axes, base_path)
            except Exception as e:
                print(f"Error submitting jobs: {e}")
                
        # Always fetch latest job list
        jobs = list_jobs(base_path)
        # Format for table
        table_data = []
        for j in jobs:
            table_data.append({
                "job_id": j.get("job_id", ""),
                "experiment_id": j.get("experiment_id", "Unknown"),
                "status": j.get("status", "unknown"),
                "submitted_at": j.get("submitted_at", "").split("T")[0] + " " + j.get("submitted_at", "").split("T")[1][:8] if "T" in j.get("submitted_at", "") else j.get("submitted_at", "")
            })
            
        return table_data

    # ---------------------------------------------------------------------------
    # Stage Debugger Callbacks
    # ---------------------------------------------------------------------------
    
    @app.callback(
        [Output("debug-data-container", "children"),
         Output("debug-artifact-info", "children")],
        Input("btn-load-artifact", "n_clicks"),
        [State("debug-experiment-select", "value"),
         State("debug-stage-select", "value")],
        prevent_initial_call=True
    )
    def load_debug_artifact(n_clicks, experiment_name, stage_name):
        """Load and display an intermediate artifact for debugging."""
        if not n_clicks or not experiment_name or not stage_name:
            raise PreventUpdate
            
        try:
            config = get_experiment(experiment_name)
            
            # Get canonical artifact name for the stage
            art_names = CANONICAL_ARTIFACT_NAMES.get(stage_name)
            if not art_names:
                return html.Div(f"No canonical artifact names defined for stage '{stage_name}'."), "Error"
                
            art_name = art_names[0] # Try the first one (e.g. 'dataset' for data_collection)
            
            artifact = make_stage_artifact(config, base_path, stage_name, art_name)
            data = load_artifact(artifact)
            
            if isinstance(data, pd.DataFrame):
                info = f"Loaded {stage_name}/{art_name}: DataFrame ({len(data)} rows × {len(data.columns)} cols)"
                
                table = dash_table.DataTable(
                    columns=df_to_dash_columns(data),
                    data=df_to_dash_records(data, max_rows=50),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                )
                return html.Div([
                    html.P("Showing first 50 rows:", className="text-muted small"),
                    table
                ]), info
                
            elif isinstance(data, dict):
                info = f"Loaded {stage_name}/{art_name}: Dictionary ({len(data)} keys)"
                # Simple display for dicts
                import json
                # Handle non-serializable objects in dict by converting to string
                safe_dict = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v for k, v in data.items()}
                return html.Pre(json.dumps(safe_dict, indent=2, default=str), style={"maxHeight": "500px", "overflowY": "auto"}), info
                
            else:
                info = f"Loaded {stage_name}/{art_name}: {type(data).__name__}"
                return html.Pre(str(data)), info
                
        except Exception as e:
            return html.Div(f"Error loading artifact: {str(e)}", className="text-danger"), "Error"
