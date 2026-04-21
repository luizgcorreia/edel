"""Callbacks for the Experiment Runner and Stage Debugger panels."""

from pathlib import Path
from dash import Dash, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import pandas as pd
from dash import dash_table

from edel.experiments.registry import get_experiment
from edel.experiments.snippets import get_snippets, save_snippet, STAGE_KEYS, STAGE_LIST
from edel.dashboard.worker import submit_job, list_jobs
from edel.io.artifact import make_stage_artifact, load_artifact, CANONICAL_ARTIFACT_NAMES
from edel.dashboard.utils import df_to_dash_columns, df_to_dash_records
import itertools
import copy
import json

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
