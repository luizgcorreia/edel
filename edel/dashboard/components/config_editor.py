"""Config Manager Component for the EDEL Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc
from edel.dashboard.utils import get_registry_options

def config_manager_layout() -> dbc.Container:
    """Layout for the Config Manager panel."""
    return dbc.Container([
        html.H3("Config Manager", className="mt-3 mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Base Config:"),
                dcc.Dropdown(
                    id="config-selector",
                    options=get_registry_options(),
                    value="scientometrics_baseline",
                    clearable=False,
                    className="mb-3"
                ),
                
                html.Label("Edit JSON:"),
                dcc.Textarea(
                    id="config-editor",
                    style={"width": "100%", "height": "500px", "fontFamily": "monospace"},
                    className="mb-3"
                ),
                
                html.Div(id="config-feedback", className="mb-3 text-danger"),
                
                html.Label("Save As:"),
                dbc.InputGroup([
                    dbc.Input(id="new-config-name", placeholder="Enter new config name..."),
                    dbc.Button("Save as New", id="btn-save-config", color="primary")
                ], className="mb-3"),
                
            ], md=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Instructions"),
                    dbc.CardBody([
                        html.P("1. Select a base configuration from the dropdown."),
                        html.P("2. Modify the JSON directly in the editor."),
                        html.P("3. Enter a new name and click 'Save as New' to register it."),
                        html.P("The new config will be available in the Experiment Runner immediately.")
                    ])
                ])
            ], md=4)
        ])
    ], fluid=True)
