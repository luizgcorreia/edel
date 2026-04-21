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
                    persistence=False,
                    className="mb-3"
                ),
                
                html.Label("Edit JSON:"),
                dcc.Textarea(
                    id="config-editor",
                    style={"width": "100%", "height": "500px", "fontFamily": "monospace"},
                    className="mb-3"
                ),
                
                html.Div(id="config-feedback", className="mb-3 text-danger"),
                
                html.Label("Save / Overwrite:"),
                dbc.InputGroup([
                    dbc.Input(id="new-config-name", placeholder="Enter new config name..."),
                    dbc.Button("Save as New", id="btn-save-config", color="primary"),
                    dbc.Button("Overwrite Current", id="btn-overwrite-config", color="warning", className="ms-1")
                ], className="mb-3"),
                
                dbc.Button([html.I(className="bi bi-trash-fill me-2"), "Delete Experiment"], 
                           id="btn-delete-config", color="danger", outline=True, className="mb-3"),
                
                # --- Deletion Confirmation Modal ---
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Confirm Deletion")),
                    dbc.ModalBody([
                        html.P("Are you sure you want to delete this experiment configuration?"),
                        html.P(id="delete-modal-exp-name", className="fw-bold text-danger"),
                        dbc.Checkbox(
                            id="delete-artifacts-checkbox",
                            label="Also delete all physical artifacts and results on disk",
                            value=False,
                        ),
                        html.Div("⚠️ This action is irreversible!", className="text-muted small mt-2")
                    ]),
                    dbc.ModalFooter([
                        dbc.Button("Cancel", id="btn-cancel-delete", className="ms-auto"),
                        dbc.Button("Confirm Delete", id="btn-confirm-delete", color="danger")
                    ]),
                ], id="delete-confirm-modal", is_open=False),
                
            ], md=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Instructions"),
                    dbc.CardBody([
                        html.P("1. Select a base configuration from the dropdown."),
                        html.P("2. Modify the JSON directly in the editor."),
                        html.P("3. 'Save as New' creates a new entry; 'Overwrite' updates the current one."),
                        html.P("4. Delete removes the config and optionally cleans up artifacts.")
                    ])
                ])
            ], md=4)
        ])
    ], fluid=True)
