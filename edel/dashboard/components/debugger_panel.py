"""Stage Debugger Component for the EDEL Dashboard."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from edel.dashboard.utils import get_registry_options
from edel.io.artifact import CANONICAL_STAGE_NAMES

def debugger_panel_layout() -> dbc.Container:
    """Layout for the Stage Debugger panel."""
    return dbc.Container([
        html.H3("Stage Debugger", className="mt-3 mb-4"),
        
        dbc.Row([
            # Left Column: Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Artifact Loader"),
                    dbc.CardBody([
                        html.Label("Experiment:"),
                        dcc.Dropdown(
                            id="debug-experiment-select",
                            options=get_registry_options(),
                            value="scientometrics_baseline",
                            clearable=False,
                            persistence=False,
                            className="mb-3"
                        ),
                        
                        html.Label("Stage:"),
                        dcc.Dropdown(
                            id="debug-stage-select",
                            options=[{"label": s, "value": s} for s in CANONICAL_STAGE_NAMES],
                            value="clustering",
                            clearable=False,
                            className="mb-3"
                        ),

                        html.Label("Anisotropy Correction Method:"),
                        dcc.Dropdown(
                            id="debug-correction-method",
                            options=[
                                {"label": "Follow Config", "value": "follow"},
                                {"label": "None", "value": "none"},
                                {"label": "Remove Top PCs", "value": "pc_removal"},
                                {"label": "Mean Centering", "value": "mean_centering"},
                            ],
                            value="follow",
                            clearable=False,
                            className="mb-3"
                        ),

                        html.Label("Top PCs to Remove (if selected):"),
                        dcc.Input(
                            id="debug-remove-pc",
                            type="number",
                            min=0,
                            max=10,
                            step=1,
                            value=0,
                            className="form-control mb-3"
                        ),
                        
                        dbc.Button("Load Artifact", id="btn-load-artifact", color="primary", className="w-100 mb-2"),
                        dbc.Button("Run Stage", id="btn-run-stage", color="success", className="w-100")
                    ])
                ])
            ], md=3),
            
            # Right Column: Data Viewer
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(id="debug-artifact-info", children="No artifact loaded"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-debug",
                            type="circle",
                            children=html.Div(id="debug-data-container")
                        )
                    ])
                ])
            ], md=9)
        ])
    ], fluid=True)
