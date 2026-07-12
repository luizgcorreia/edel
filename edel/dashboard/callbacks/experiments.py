"""Callbacks for the Experiment Runner and Stage Debugger panels."""

from pathlib import Path
from dash import Dash, Input, Output, State, callback_context, html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
from dash import dash_table

from edel.experiments.registry import get_experiment
from edel.experiments.snippets import get_snippets, save_snippet, STAGE_KEYS, STAGE_LIST
from edel.dashboard.worker import submit_job, list_jobs, get_job_log
from edel.io.artifact import make_stage_artifact, load_artifact, save_artifact, CANONICAL_ARTIFACT_NAMES
from edel.dashboard.utils import df_to_dash_columns, df_to_dash_records
import edel.pipeline as pipeline
import edel.viz as viz
import itertools
import copy
import json
import io
import base64
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def split_filter_part(filter_part):
    for operator in [['eq', '=='],
                     ['ne', '!='],
                     ['lt', '<'],
                     ['le', '<='],
                     ['gt', '>'],
                     ['ge', '>='],
                     ['contains'],
                     ['datestartswith']]:
        if len(operator) == 2:
            operator_name, operator_string = operator
        else:
            operator_name = operator[0]
            operator_string = operator[0]
        if ' ' + operator_name + ' ' in filter_part:
            return filter_part.split(' ' + operator_name + ' ', 1) + [operator_string]
    return [None]

def parse_filter_query(filter_query, df):
    if not filter_query:
        return df
    
    current_df = df
    for part in filter_query.split(' && '):
        split_part = split_filter_part(part)
        if len(split_part) == 3:
            col_name, val, operator = split_part
            col_name = col_name.replace('{', '').replace('}', '').strip()
            val = val.strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
                
            try:
                if operator == 'contains':
                    current_df = current_df[current_df[col_name].astype(str).str.contains(val, case=False, na=False)]
                elif operator == 'datestartswith':
                    current_df = current_df[current_df[col_name].astype(str).str.startswith(val)]
                elif operator == '==':
                    current_df = current_df[current_df[col_name].astype(str) == val]
                elif operator == '!=':
                    current_df = current_df[current_df[col_name].astype(str) != val]
                else:
                    val_num = float(val)
                    if operator == '<':
                        current_df = current_df[current_df[col_name] < val_num]
                    elif operator == '<=':
                        current_df = current_df[current_df[col_name] <= val_num]
                    elif operator == '>':
                        current_df = current_df[current_df[col_name] > val_num]
                    elif operator == '>=':
                        current_df = current_df[current_df[col_name] >= val_num]
            except Exception as e:
                print(f"Error applying filter part '{part}': {e}")
                
    return current_df

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

    @app.callback(
        [Output("selected-job-info", "children"),
         Output("job-log-display", "children"),
         Output("btn-cancel-job", "disabled"),
         Output("btn-delete-job", "disabled")],
        [Input("job-queue-table", "selected_rows"),
         Input("job-queue-interval", "n_intervals")],
        [State("job-queue-table", "data")]
    )
    def update_job_logs(selected_rows, n_intervals, table_data):
        """Display logs and details for the selected job in the table."""
        if not selected_rows or not table_data:
            return "No job selected", "Select a job from the table to view its execution logs.", True, True
            
        row_idx = selected_rows[0]
        if row_idx >= len(table_data):
            return "No job selected", "Select a job from the table to view its execution logs.", True, True
            
        job_info = table_data[row_idx]
        job_id = job_info.get("job_id")
        status = job_info.get("status")
        experiment_id = job_info.get("experiment_id")
        
        log_text = get_job_log(job_id, base_path, tail=100)
        if not log_text:
            log_text = "No log messages generated yet or log file does not exist."
            
        header = f"Job: {job_id} ({experiment_id}) — Status: {status.upper()}"
        cancel_disabled = (status not in ["running", "pending"])
        delete_disabled = (status not in ["done", "failed"])
        return header, log_text, cancel_disabled, delete_disabled

    @app.callback(
        [Output("job-queue-table", "selected_rows", allow_duplicate=True),
         Output("job-queue-table", "data", allow_duplicate=True)],
        Input("btn-cancel-job", "n_clicks"),
        [State("job-queue-table", "selected_rows"),
         State("job-queue-table", "data")],
        prevent_initial_call=True
    )
    def handle_cancel_job(n_clicks, selected_rows, table_data):
        if not n_clicks or not selected_rows or not table_data:
            raise PreventUpdate
            
        row_idx = selected_rows[0]
        if row_idx >= len(table_data):
            raise PreventUpdate
            
        job_info = table_data[row_idx]
        job_id = job_info.get("job_id")
        
        from edel.dashboard.worker import cancel_job
        cancel_job(job_id, base_path)
        
        # Trigger queue refresh
        from edel.dashboard.worker import list_jobs
        jobs = list_jobs(base_path)
        new_table_data = []
        for j in jobs:
            new_table_data.append({
                "job_id": j.get("job_id", ""),
                "experiment_id": j.get("experiment_id", "Unknown"),
                "status": j.get("status", "unknown"),
                "submitted_at": j.get("submitted_at", "").split("T")[0] + " " + j.get("submitted_at", "").split("T")[1][:8] if "T" in j.get("submitted_at", "") else j.get("submitted_at", "")
            })
            
        return [], new_table_data

    @app.callback(
        [Output("job-queue-table", "selected_rows", allow_duplicate=True),
         Output("job-queue-table", "data", allow_duplicate=True)],
        Input("btn-delete-job", "n_clicks"),
        [State("job-queue-table", "selected_rows"),
         State("job-queue-table", "data")],
        prevent_initial_call=True
    )
    def handle_delete_job(n_clicks, selected_rows, table_data):
        if not n_clicks or not selected_rows or not table_data:
            raise PreventUpdate
            
        row_idx = selected_rows[0]
        if row_idx >= len(table_data):
            raise PreventUpdate
            
        job_info = table_data[row_idx]
        job_id = job_info.get("job_id")
        
        from edel.dashboard.worker import delete_job_record
        delete_job_record(job_id, base_path)
        
        # Trigger queue refresh
        from edel.dashboard.worker import list_jobs
        jobs = list_jobs(base_path)
        new_table_data = []
        for j in jobs:
            new_table_data.append({
                "job_id": j.get("job_id", ""),
                "experiment_id": j.get("experiment_id", "Unknown"),
                "status": j.get("status", "unknown"),
                "submitted_at": j.get("submitted_at", "").split("T")[0] + " " + j.get("submitted_at", "").split("T")[1][:8] if "T" in j.get("submitted_at", "") else j.get("submitted_at", "")
            })
            
        return [], new_table_data

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
         Output("debug-artifact-info", "children"),
         Output("artifact-update-store", "data", allow_duplicate=True)],
        [Input("btn-load-artifact", "n_clicks"),
         Input("btn-run-stage", "n_clicks")],
        [State("debug-experiment-select", "value"),
         State("debug-stage-select", "value"),
         State("config-editor", "value"),
         State("debug-correction-method", "value"),
         State("debug-remove-pc", "value")],
        prevent_initial_call=True
    )
    def handle_debug_action(load_clicks, run_clicks, experiment_name, stage_name, config_json, correction_method, remove_pc):
        """Load or Run an intermediate stage for debugging."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if not experiment_name or not stage_name:
            return None, None, time.time()
            
        try:
            # Prefer the config currently in the editor UI to avoid hash mismatches
            if config_json:
                from edel.dashboard.utils import parse_config_json
                config = parse_config_json(config_json)
                if config is None:
                    raise ValueError("Invalid JSON in config editor.")
            else:
                config = get_experiment(experiment_name)
            
            # Resolve anisotropy correction parameters
            dr_cfg = config.get("dimensionality_reduction", {})
            if correction_method == "follow":
                remove_pc_resolved = dr_cfg.get("remove_top_pcs", 0)
                method_resolved = dr_cfg.get("anisotropy_method", "pc_removal" if remove_pc_resolved > 0 else "none")
            else:
                method_resolved = correction_method
                remove_pc_resolved = remove_pc

            # Inject resolved parameters into config ONLY if we are overriding 'Follow Config'
            # This ensures that 'Follow Config' matches the hash of the original experiment perfectly.
            if correction_method != "follow":
                if method_resolved != "none":
                    if "dimensionality_reduction" not in config:
                        config["dimensionality_reduction"] = {}
                    config["dimensionality_reduction"]["anisotropy_method"] = method_resolved
                    config["dimensionality_reduction"]["remove_top_pcs"] = remove_pc_resolved
                elif "dimensionality_reduction" in config:
                    # If we explicitly select 'None' in the UI, we should probably ensure 
                    # the config reflects that, but only if it's an override.
                    config["dimensionality_reduction"]["anisotropy_method"] = "none"
                    config["dimensionality_reduction"]["remove_top_pcs"] = 0

            if triggered_id == "btn-run-stage":
                # --- RUN STAGE LOGIC (Mirroring exploration.ipynb) ---
                data = None
                field = None
                report = None
                
                # 1. Load dependencies
                if stage_name == "data_collection":
                    data, report = pipeline.run_data_stage(config)
                elif stage_name == "structured_abstracts":
                    prev_art = make_stage_artifact(config, base_path, "data_collection", "dataset")
                    data, report = pipeline.run_structuring_stage(load_artifact(prev_art), config)
                elif stage_name == "embeddings":
                    prev_art = make_stage_artifact(config, base_path, "structured_abstracts", "sa")
                    prev_data = load_artifact(prev_art)
                    if isinstance(prev_data, tuple):
                        prev_data = prev_data[0]
                    res = pipeline.run_embedding_stage(prev_data, config, return_report=True)
                    if isinstance(res, tuple):
                        data, report = res
                    else:
                        data = res
                elif stage_name == "dimensionality_reduction":
                    prev_art = make_stage_artifact(config, base_path, "embeddings", "embeddings")
                    data, report = pipeline.run_projection_stage(load_artifact(prev_art), config)
                elif stage_name == "vector_field":
                    prev_art = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
                    field = pipeline.run_vector_field_stage(load_artifact(prev_art), config)
                    data = field # For generic display
                elif stage_name == "clustering":
                    art_df = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
                    art_field = make_stage_artifact(config, base_path, "vector_field", "vf")
                    res = pipeline.run_clustering_stage(load_artifact(art_df), load_artifact(art_field), config)
                    if isinstance(res, tuple) and len(res) == 3:
                        data, field, report = res
                    else:
                        data, field = res
                elif stage_name == "labeling":
                    art_df = make_stage_artifact(config, base_path, "clustering", "clustering")
                    art_field = make_stage_artifact(config, base_path, "clustering", "field_clustering")
                    from edel.io.llm import get_llm_client
                    llm_client = get_llm_client(config.get("labeling", {}))
                    data = pipeline.run_labeling_stage(load_artifact(art_df), load_artifact(art_field), config, llm_client)
                elif stage_name == "output":
                    art_df = make_stage_artifact(config, base_path, "clustering", "clustering")
                    art_field = make_stage_artifact(config, base_path, "vector_field", "vf")
                    data = pipeline.run_landscape_stage(load_artifact(art_df), load_artifact(art_field), config)
                
                # 2. Save artifacts
                art_names = CANONICAL_ARTIFACT_NAMES.get(stage_name, [])
                saved_info = ""
                if stage_name == "clustering" and isinstance(data, pd.DataFrame) and isinstance(field, pd.DataFrame):
                    p1 = save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[0]), data)
                    p2 = save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[1]), field)
                    if report is not None and len(art_names) > 2:
                        save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[2]), report)
                    saved_info = str(p1.parent)
                elif data is not None:
                    p = save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[0]), data)
                    saved_info = str(p)
                    # Save report if it was generated
                    if report is not None and len(art_names) > 1:
                        save_artifact(make_stage_artifact(config, base_path, stage_name, art_names[1]), report)
                
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
                
                # Load report if available for this stage
                report = None
                if (stage_name == "structured_abstracts" or stage_name == "data_collection" or stage_name == "dimensionality_reduction" or stage_name == "embeddings") and len(art_names) > 1:
                    try:
                        report_art = make_stage_artifact(config, base_path, stage_name, art_names[1])
                        report = load_artifact(report_art)
                    except: pass

            # --- VISUALIZATION LOGIC ---
            viz_components = []
            
            if stage_name == "data_collection":
                if report:
                    # Show report as a preformatted JSON block
                    viz_components.append(html.H5("Harvest Filtering Report", className="mt-3"))
                    viz_components.append(html.Pre(json.dumps(report, indent=2), className="p-3 bg-dark text-white rounded small"))
                    
                    # Add filtering stats plot
                    viz_components.append(capture_matplotlib_plot(viz.plot_filtering_stats, report))

                viz_components.append(capture_matplotlib_plot(viz.plot_publication_year_dist, data))
                viz_components.append(capture_matplotlib_plot(viz.plot_citation_dist, data))
                viz_components.append(capture_matplotlib_plot(viz.plot_abstract_length_dist, data))
            
            elif stage_name == "structured_abstracts":
                if report:
                    # Show report as a preformatted JSON block
                    viz_components.append(html.H5("Structuring Report", className="mt-3"))
                    viz_components.append(html.Pre(json.dumps(report, indent=2), className="p-3 bg-dark text-white rounded small"))
                    
                    # Add charts
                    viz_components.append(capture_matplotlib_plot(viz.plot_segmentation_stats, report))
                else:
                    viz_components.append(html.Div("No structuring report found for this artifact.", className="text-muted small"))
                    
                if "language" in data.columns:
                    viz_components.append(capture_matplotlib_plot(viz.plot_language_dist, data))
            
            elif stage_name == "embeddings":
                if report and "aspect_coverage" in report:
                    card_body = [
                        html.P(f"Initial: {report.get('initial_count', 0):,} → Final: {report.get('final_count', 0):,} ({report.get('total_filtered', 0):,} filtered)", className="fw-bold mb-2")
                    ]
                    
                    rows = []
                    for aspect, cov in report["aspect_coverage"].items():
                        filtered = cov.get("filtered", 0)
                        stayed = cov.get("stayed", 0)
                        total = filtered + stayed
                        pct = (stayed / total * 100) if total > 0 else 0
                        rows.append(html.Tr([
                            html.Td(aspect.capitalize(), style={"fontWeight": "bold"}),
                            html.Td(f"{filtered:,}"),
                            html.Td(f"{stayed:,}"),
                            html.Td(f"{pct:.1f}%")
                        ]))
                    
                    tbl = html.Table(
                        [html.Thead(html.Tr([html.Th("Aspect"), html.Th("Filtered"), html.Th("Stayed"), html.Th("Coverage Ratio")]))] + 
                        [html.Tbody(rows)],
                        className="table table-sm table-hover table-striped small border mb-3",
                        style={"maxWidth": "600px"}
                    )
                    card_body.append(tbl)
                    
                    card_body.append(dbc.Accordion([
                        dbc.AccordionItem(
                            html.Pre(json.dumps(report, indent=2), className="p-3 bg-dark text-white rounded small"),
                            title="View Raw JSON Report"
                        )
                    ], start_collapsed=True, className="mb-4"))
                    
                    viz_components.append(html.H5("Aspect Coverage Summary", className="mt-3"))
                    viz_components.append(html.Div(card_body))

                from edel.experiments.metrics.embedding import embedding_metrics
                from edel.pipeline.projection import detect_embedding_dimensions
                n_dims = detect_embedding_dimensions(data, config)
                m_res = embedding_metrics(
                    {"embedding": data, "_dimensions": n_dims}, 
                    correction_method=method_resolved or "none",
                    remove_pc=remove_pc_resolved or 0
                )
                
                metrics = m_res.get("metrics", {})
                if metrics:
                    method_label = "None" if method_resolved == "none" else ("Mean Centering" if method_resolved == "mean_centering" else f"PC Removal (N={remove_pc_resolved})")
                    viz_components.append(html.H5(f"Embedding Metrics (Method: {method_label})", className="mt-3"))
                    
                    # Create a simple table for metrics
                    rows = [
                        html.Tr([html.Td(k, style={"fontWeight": "bold"}), html.Td(f"{v:.4f}")])
                        for k, v in metrics.items()
                    ]
                    table = html.Table(
                        [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))] + 
                        [html.Tbody(rows)],
                        className="table table-sm table-hover table-striped small border",
                        style={"maxWidth": "500px"}
                    )
                    viz_components.append(table)
                else:
                    viz_components.append(html.Div("No embedding metrics could be computed.", className="text-muted small"))

            elif stage_name == "dimensionality_reduction":
                if report and "evals" in report:
                    viz_components.append(html.H5("Diffusion Map Eigenvalues", className="mt-3"))
                    viz_components.append(capture_matplotlib_plot(viz.plot_diffusion_eigenvalues, report["evals"]))
                    
                method = config.get("dimensionality_reduction", {}).get("method", "umap")
                viz_components.append(capture_matplotlib_plot(viz.plot_projection_2d, data, method=method, aspect="problem"))
                viz_components.append(capture_matplotlib_plot(viz.plot_projection_2d, data, method=method, aspect="method"))
                viz_components.append(capture_matplotlib_plot(viz.plot_projection_2d, data, method=method, aspect="finding"))
                viz_components.append(capture_matplotlib_plot(viz.plot_projection_2d, data, method=method, aspect="interpretation"))
                viz_components.append(capture_matplotlib_plot(viz.plot_paper_style_pca, data))
                
                # Transition space plot
                from edel.pipeline.projection import detect_embedding_dimensions
                n_dims = detect_embedding_dimensions(data, config)
                viz_components.append(capture_matplotlib_plot(
                    viz.plot_epistemic_transition_space, 
                    data, 
                    quantile=0.95, 
                    dimensions=n_dims,
                    correction_method=method_resolved or "none",
                    remove_pc=remove_pc_resolved or 0
                ))

                # Unified Discourse Space plot
                viz_components.append(capture_matplotlib_plot(
                    viz.plot_unified_discourse_space,
                    data,
                    method=method,
                    dimensions=n_dims,
                    correction_method=method_resolved or "none",
                    remove_pc=remove_pc_resolved or 0
                ))

                # Calculate overlap metrics
                from edel.experiments.metrics.embedding import embedding_metrics
                m_res = embedding_metrics(
                    {"embedding": data, "_dimensions": n_dims}, 
                    correction_method=method_resolved or "none",
                    remove_pc=remove_pc_resolved or 0
                )
                metrics = m_res.get("metrics", {})
                if metrics:
                    viz_components.append(html.H5("Unified Discourse Space & Transition Space Overlap Metrics", className="mt-4"))
                    rows = [
                        html.Tr([html.Td("Aspect Space: Silhouette Score", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_aspect_silhouette', 0.0):.4f}")]),
                        html.Tr([html.Td("Aspect Space: 1-NN Aspect Classification Accuracy", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_aspect_accuracy_1nn', 0.0):.1%}")]),
                        html.Tr([html.Td("Aspect Space: NN Same-Paper Ratio (Paper identity)", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_aspect_nn_same_paper', 0.0):.1%}")]),
                        html.Tr([html.Td("Aspect Space: NN Same-Aspect Ratio (Aspect identity)", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_aspect_nn_same_category', 0.0):.1%}")]),
                        html.Tr([html.Td("Aspect Space: NN Other Ratio", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_aspect_nn_other', 0.0):.1%}")]),
                        
                        html.Tr([html.Td(html.Hr(), colSpan=2)]),
                        
                        html.Tr([html.Td("Transition Space: Silhouette Score", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_trans_silhouette', 0.0):.4f}")]),
                        html.Tr([html.Td("Transition Space: 1-NN Transition Classification Accuracy", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_trans_accuracy_1nn', 0.0):.1%}")]),
                        html.Tr([html.Td("Transition Space: NN Same-Paper Ratio (Paper identity)", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_trans_nn_same_paper', 0.0):.1%}")]),
                        html.Tr([html.Td("Transition Space: NN Same-Transition Ratio (Transition identity)", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_trans_nn_same_category', 0.0):.1%}")]),
                        html.Tr([html.Td("Transition Space: NN Other Ratio", style={"fontWeight": "bold"}), html.Td(f"{metrics.get('joint_trans_nn_other', 0.0):.1%}")]),
                    ]
                    table = html.Table(
                        [html.Thead(html.Tr([html.Th("Metric Description"), html.Th("Value")]))] + 
                        [html.Tbody(rows)],
                        className="table table-sm table-hover table-striped small border",
                        style={"maxWidth": "650px", "marginBottom": "20px"}
                    )
                    viz_components.append(table)
            
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

            if isinstance(data, pd.DataFrame):
                info = f"{info_prefix}: DataFrame ({len(data)} rows × {len(data.columns)} cols)"
                table = dash_table.DataTable(
                    id="debug-data-table",
                    columns=df_to_dash_columns(data),
                    data=[],  # Will be populated dynamically by the callback
                    page_current=0,
                    page_size=10,
                    page_action="custom",
                    filter_action="custom",
                    filter_query="",
                    sort_action="custom",
                    sort_by=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': '180px',
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'backgroundColor': '#f8f9fa',
                        'border': '1px solid #dee2e6'
                    },
                    style_data={
                        'border': '1px solid #dee2e6'
                    },
                    tooltip_delay=250,
                    tooltip_duration=None,
                )
                return html.Div([
                    html.Div(viz_components),
                    html.Hr(),
                    html.H5("Artifact Data Explorer", className="mb-2"),
                    table,
                    html.Div(id="debug-row-detail-viewer", className="mt-3")
                ]), info, time.time()
                
            elif isinstance(data, dict):
                info = f"{info_prefix}: Dictionary ({len(data)} keys)"
                safe_dict = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v for k, v in data.items()}
                return html.Div([
                    html.Div(viz_components),
                    html.Hr(),
                    html.Pre(json.dumps(safe_dict, indent=2, default=str), style={"maxHeight": "500px", "overflowY": "auto"})
                ]), info, time.time()
                
            else:
                info = f"{info_prefix}: {type(data).__name__}"
                return html.Div([
                    html.Div(viz_components),
                    html.Hr(),
                    html.Pre(str(data))
                ]), info, time.time()
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return dbc.Alert([
                html.H4("Stage Execution Error", className="alert-heading"),
                html.P(f"An error occurred while running or loading the stage '{stage_name}':"),
                html.Hr(),
                html.Pre(str(e), className="mb-0 small")
            ], color="danger", className="mt-3"), f"Error in {stage_name}", time.time()

    @app.callback(
        [Output("debug-data-table", "data"),
         Output("debug-data-table", "page_count"),
         Output("debug-data-table", "tooltip_data")],
        [Input("debug-data-table", "page_current"),
         Input("debug-data-table", "page_size"),
         Input("debug-data-table", "sort_by"),
         Input("debug-data-table", "filter_query")],
        [State("debug-experiment-select", "value"),
         State("debug-stage-select", "value"),
         State("config-editor", "value"),
         State("debug-correction-method", "value"),
         State("debug-remove-pc", "value")]
    )
    def update_table_data(page_current, page_size, sort_by, filter_query, experiment_name, stage_name, config_json, correction_method, remove_pc):
        if not experiment_name or not stage_name:
            raise PreventUpdate
            
        try:
            # Reconstruct the config object
            if config_json:
                from edel.dashboard.utils import parse_config_json
                config = parse_config_json(config_json)
            else:
                config = get_experiment(experiment_name)
                
            if not config:
                raise PreventUpdate
                
            # Resolve anisotropy correction parameters (same as in handle_debug_action)
            if correction_method != "follow":
                if "dimensionality_reduction" not in config:
                    config["dimensionality_reduction"] = {}
                config["dimensionality_reduction"]["anisotropy_method"] = correction_method
                config["dimensionality_reduction"]["remove_top_pcs"] = remove_pc
                
            # Load artifact from disk
            art_names = CANONICAL_ARTIFACT_NAMES.get(stage_name, [])
            if not art_names:
                raise PreventUpdate
            art_name = art_names[0]
            artifact = make_stage_artifact(config, base_path, stage_name, art_name)
            
            # Check if artifact exists on disk before loading
            if not (artifact.parquet_path.exists() or artifact.pkl_path.exists()):
                return [], 0, []
                
            data = load_artifact(artifact)
            if isinstance(data, tuple) and len(data) > 0 and isinstance(data[0], pd.DataFrame):
                data = data[0]
                
            if not isinstance(data, pd.DataFrame):
                raise PreventUpdate
                
            # 1. Apply filtering
            filtered_df = parse_filter_query(filter_query, data)
            
            # 2. Apply sorting
            if sort_by and len(sort_by) > 0:
                by_cols = [s['column_id'] for s in sort_by]
                ascending_flags = [s['direction'] == 'asc' for s in sort_by]
                filtered_df = filtered_df.sort_values(by=by_cols, ascending=ascending_flags)
                
            # 3. Paginate
            total_rows = len(filtered_df)
            page_count = (total_rows + page_size - 1) // page_size
            
            start_idx = page_current * page_size
            end_idx = start_idx + page_size
            paginated_df = filtered_df.iloc[start_idx:end_idx]
            
            # Convert to records
            records = df_to_dash_records(paginated_df)
            
            # 4. Generate tooltips
            tooltip_data = [
                {
                    col: {'value': str(val), 'type': 'markdown'}
                    for col, val in row.items() if val is not None
                }
                for row in records
            ]
            
            return records, page_count, tooltip_data
            
        except Exception as e:
            print(f"Error in update_table_data: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, []

    @app.callback(
        Output("debug-row-detail-viewer", "children"),
        [Input("debug-data-table", "active_cell")],
        [State("debug-data-table", "data")]
    )
    def display_row_details(active_cell, table_data):
        if not active_cell or not table_data:
            return html.Div(
                "Click on any cell in the table to view its full details here.",
                className="text-muted italic my-3 text-center small border p-3 rounded"
            )
            
        row_idx = active_cell['row']
        if row_idx >= len(table_data):
            return html.Div("Row index out of bounds.", className="text-danger small")
            
        row_data = table_data[row_idx]
        
        # Build a beautiful bootstrap details layout
        cards = []
        
        # Special layouts for aspects
        aspects = ["problem", "method", "finding", "interpretation"]
        has_aspects = any(a in row_data for a in aspects)
        
        if has_aspects:
            # Map internal aspect names to the Isabelle terms:
            # problem -> Premises
            # method -> Skeleton
            # finding -> Tactics
            # interpretation -> Conclusion
            aspect_mapping = {
                "problem": ("Premises", "info"),
                "method": ("Skeleton", "secondary"),
                "finding": ("Tactics", "success"),
                "interpretation": ("Conclusion", "primary")
            }
            
            aspect_cards = []
            for aspect in aspects:
                val = row_data.get(aspect, "none")
                title, color = aspect_mapping[aspect]
                if not val or str(val).strip() == "":
                    val = "none"
                    
                aspect_cards.append(dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(title, className=f"bg-{color} text-white py-1 fw-bold small"),
                        dbc.CardBody(
                            html.Pre(str(val), className="mb-0 small", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"}),
                            style={"maxHeight": "200px", "overflowY": "auto", "padding": "10px"}
                        )
                    ], className="mb-3 shadow-sm"),
                    md=6
                ))
            cards.append(html.H6("Epistemic Aspects (I/L Isabelle RAG System)", className="mt-2 mb-3 text-muted"))
            cards.append(dbc.Row(aspect_cards))
            
        # Display other key-value metadata in a table or list
        metadata_rows = []
        for k, v in row_data.items():
            if k not in aspects and v is not None:
                val_str = str(v)
                if len(val_str) > 1000:
                    val_str = val_str[:1000] + " ... [truncated]"
                metadata_rows.append(html.Tr([
                    html.Td(html.Strong(k), style={"width": "20%"}),
                    html.Td(html.Pre(val_str, className="mb-0", style={"whiteSpace": "pre-wrap", "fontSize": "0.85em"}))
                ]))
                
        if metadata_rows:
            cards.append(html.H6("Entry Metadata & Attributes", className="mt-2 mb-3 text-muted"))
            cards.append(html.Table(
                [html.Tbody(metadata_rows)],
                className="table table-sm table-striped border small shadow-sm"
            ))
            
        return html.Div(cards, className="p-3 border rounded bg-light")
