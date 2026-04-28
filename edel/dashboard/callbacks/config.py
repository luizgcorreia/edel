"""Callbacks for the Config Manager panel."""

import dash
from dash import Dash, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from edel.dashboard.utils import config_to_json, parse_config_json, get_registry_options
from edel.experiments.registry import register_experiment, list_experiments

def register_config_callbacks(app: Dash) -> None:
    
    # --- Global Registry Sync ---
    @app.callback(
        Output("experiment-store", "data", allow_duplicate=True),
        Input("config-selector", "id"), # Trigger once on load
        prevent_initial_call="initial_duplicate"
    )
    def init_registry_store(_):
        return get_registry_options()

    @app.callback(
        [Output("map-experiment-select", "options"),
         Output("debug-experiment-select", "options"),
         Output("sweep-base-config", "options")],
        Input("experiment-store", "data"),
        prevent_initial_call=False
    )
    def sync_all_dropdowns(options):
        if not options:
            return [dash.no_update] * 3
        return [options] * 3

    # --- Value Synchronization ---
    @app.callback(
        [Output("config-selector", "value", allow_duplicate=True),
         Output("map-experiment-select", "value", allow_duplicate=True),
         Output("debug-experiment-select", "value", allow_duplicate=True)],
        [Input("config-selector", "value"),
         Input("map-experiment-select", "value"),
         Input("debug-experiment-select", "value")],
        prevent_initial_call=True
    )
    def sync_experiment_selection(v1, v2, v3):
        """Ensure all experiment dropdowns show the same selected value."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        new_value = None
        if triggered_id == "config-selector": new_value = v1
        elif triggered_id == "map-experiment-select": new_value = v2
        elif triggered_id == "debug-experiment-select": new_value = v3
        
        if not new_value:
            raise PreventUpdate
            
        return [new_value] * 3

    
    @app.callback(
        [Output("config-editor", "value"),
         Output("config-store", "data")],
        Input("config-selector", "value")
    )
    def load_config(experiment_name: str):
        """Load selected config into editor and store."""
        if not experiment_name:
            raise PreventUpdate
            
        json_str = config_to_json(experiment_name)
        config_dict = parse_config_json(json_str)
        return json_str, config_dict

    @app.callback(
        [Output("config-feedback", "children"),
         Output("config-selector", "options"),
         Output("config-selector", "value"),
         Output("delete-confirm-modal", "is_open"),
         Output("delete-modal-exp-name", "children"),
         Output("experiment-store", "data", allow_duplicate=True)],
        [Input("btn-save-config", "n_clicks"),
         Input("btn-overwrite-config", "n_clicks"),
         Input("btn-delete-config", "n_clicks"),
         Input("btn-cancel-delete", "n_clicks"),
         Input("btn-confirm-delete", "n_clicks")],
        [State("config-selector", "value"),
         State("new-config-name", "value"),
         State("config-editor", "value"),
         State("delete-artifacts-checkbox", "value"),
         State("delete-confirm-modal", "is_open")],
        prevent_initial_call=True
    )
    def manage_config(n_save, n_over, n_del, n_cancel, n_confirm, 
                      current_name, new_name, config_json_str, 
                      delete_artifacts, is_modal_open):
        """Manage registration, overwriting, and deletion of configurations."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # --- Handle Modal Toggle ---
        if triggered_id == "btn-delete-config":
            return dash.no_update, dash.no_update, dash.no_update, True, current_name, dash.no_update
        if triggered_id == "btn-cancel-delete":
            return dash.no_update, dash.no_update, dash.no_update, False, "", dash.no_update
            
        # --- Handle Deletion ---
        if triggered_id == "btn-confirm-delete":
            try:
                from edel.experiments.registry import delete_experiment, list_experiments
                from edel.io.artifact import delete_experiment_artifacts
                from pathlib import Path
                
                # 1. Optionally delete artifacts
                cleanup_msg = ""
                if delete_artifacts:
                    config_dict = parse_config_json(config_json_str)
                    if config_dict:
                        count = delete_experiment_artifacts(config_dict, Path("artifacts"))
                        cleanup_msg = f" and {count} artifacts"
                
                # 2. Delete from registry
                delete_experiment(current_name)
                
                options = get_registry_options()
                new_val = options[0]["value"] if options else None
                
                return f"Successfully deleted '{current_name}'{cleanup_msg}.", options, new_val, False, "", options
            except Exception as e:
                return f"Error deleting config: {e}", dash.no_update, dash.no_update, False, "", dash.no_update

        # --- Handle Save/Overwrite ---
        target_name = new_name.strip() if triggered_id == "btn-save-config" else current_name
        if not target_name:
            return "Please provide a name.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        config_dict = parse_config_json(config_json_str)
        if config_dict is None:
            return "Invalid JSON. Please fix errors before saving.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        try:
            from edel.experiments.registry import register_experiment
            register_experiment(target_name, config_dict)
            options = get_registry_options()
            return f"Successfully saved '{target_name}'!", options, target_name, dash.no_update, dash.no_update, options
        except Exception as e:
            return f"Error saving config: {e}", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
