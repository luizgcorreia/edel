"""Callbacks for the Config Manager panel."""

import dash
from dash import Dash, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from edel.dashboard.utils import config_to_json, parse_config_json, get_registry_options
from edel.experiments.registry import register_experiment

def register_config_callbacks(app: Dash) -> None:
    
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
         Output("config-selector", "options")],
        Input("btn-save-config", "n_clicks"),
        State("new-config-name", "value"),
        State("config-editor", "value"),
        prevent_initial_call=True
    )
    def save_new_config(n_clicks, new_name, config_json_str):
        """Parse JSON and register as a new experiment config."""
        if not n_clicks or not new_name or not config_json_str:
            return "Please provide a name and valid JSON.", dash.no_update
            
        new_name = new_name.strip()
        if not new_name:
            return "Name cannot be empty.", dash.no_update
            
        config_dict = parse_config_json(config_json_str)
        if config_dict is None:
            return "Invalid JSON. Please fix errors before saving.", dash.no_update
            
        try:
            register_experiment(new_name, config_dict)
            return f"Successfully saved '{new_name}'!", get_registry_options()
        except Exception as e:
            return f"Error saving config: {e}", dash.no_update
