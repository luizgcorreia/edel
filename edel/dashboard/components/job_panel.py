"""Experiment Runner Component for the EDEL Dashboard."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from edel.dashboard.utils import get_default_experiment, get_registry_options
from edel.experiments.snippets import get_snippets, STAGE_LIST

def _build_snippet_row(stage_name: str) -> dbc.Row:
    """Helper to build a sweep axis row for a specific stage."""
    snippets = get_snippets(stage_name)
    options = [{"label": name, "value": name} for name in snippets.keys()]
    
    # ID is derived from stage name (e.g. "Embeddings" -> "sweep-embeddings")
    id_name = f"sweep-{stage_name.lower().replace(' ', '-')}"
    
    return dbc.Row([
        dbc.Col(dbc.Label(stage_name), width=4),
        dbc.Col(dcc.Dropdown(
            id=id_name,
            options=options,
            multi=True,
            placeholder=f"Select {stage_name} variants...",
            className="mb-2"
        ), width=8)
    ], className="mb-1 align-items-center")

def job_panel_layout() -> dbc.Container:
    """Layout for the Experiment Runner (Job Queue) panel."""
    return dbc.Container([
        html.H3("Experiment Runner", className="mt-3 mb-4"),
        
        dbc.Row([
            # Left Column: Sweep Builder
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sweep Builder"),
                    dbc.CardBody([
                        html.Label("Base Config:"),
                        dcc.Dropdown(
                            id="sweep-base-config",
                            options=get_registry_options(),
                            value=get_default_experiment(),
                            clearable=False,
                            persistence=False,
                            className="mb-3"
                        ),
                        
                        html.H6("Sweep Axes", className="mt-3"),
                        html.Div(id="sweep-axes-container", children=[
                            _build_snippet_row(stage) for stage in STAGE_LIST
                        ]),
                        
                        html.Div([
                            dbc.Button("＋ New Snippet", id="btn-open-snippet-modal", size="sm", color="info", outline=True, className="mt-2"),
                        ], className="text-end"),

                        html.Hr(),
                        html.Div(id="sweep-summary", className="mb-3 p-2 bg-light border rounded small"),
                        dbc.Button("Submit Sweep", id="btn-submit-jobs", color="success", className="w-100"),
                        
                        # Modal for creating new snippets
                        dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Save Config Snippet")),
                            dbc.ModalBody([
                                dbc.Label("Stage:"),
                                dcc.Dropdown(
                                    id="snippet-stage-select",
                                    options=[{"label": s, "value": s} for s in [
                                        "Structured Abstracts", "Embeddings", "Projection", 
                                        "Vector Field", "Clustering", "Labeling", "Landscape"
                                    ]],
                                    className="mb-3"
                                ),
                                dbc.Label("Snippet Name (e.g. 'gpt-4o-mini', 'umap_15n'):"),
                                dbc.Input(id="snippet-name-input", placeholder="Enter name...", className="mb-3"),
                                dbc.Label("Config JSON:"),
                                dcc.Textarea(
                                    id="snippet-config-input",
                                    placeholder='{ "model": "..." }',
                                    style={"width": "100%", "height": "200px", "fontFamily": "monospace"}
                                ),
                            ]),
                            dbc.ModalFooter(
                                dbc.Button("Save Snippet", id="btn-save-snippet", color="primary")
                            ),
                        ], id="snippet-modal", is_open=False),
                    ])
                ])
            ], md=4),
            
            # Right Column: Job Queue
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col("Job Queue"),
                            dbc.Col([
                                dbc.Button("Cancel Job", id="btn-cancel-job", size="sm", color="danger", outline=True, className="me-2", disabled=True),
                                dbc.Button("Delete Job", id="btn-delete-job", size="sm", color="warning", outline=True, className="me-2", disabled=True),
                                dbc.Button("Refresh", id="btn-refresh-jobs", size="sm", color="secondary", outline=True),
                            ], className="text-end")
                        ])
                    ),
                    dbc.CardBody([
                        dcc.Interval(id='job-queue-interval', interval=5000, n_intervals=0),
                        dash_table.DataTable(
                            id='job-queue-table',
                            columns=[
                                {"name": "Job ID", "id": "job_id"},
                                {"name": "Experiment", "id": "experiment_id"},
                                {"name": "Status", "id": "status"},
                                {"name": "Submitted", "id": "submitted_at"}
                            ],
                            data=[],
                            row_selectable="single",
                            selected_rows=[],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{status} = "done"'}, 'backgroundColor': '#d4edda', 'color': '#155724'},
                                {'if': {'filter_query': '{status} = "failed"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
                                {'if': {'filter_query': '{status} = "running"'}, 'backgroundColor': '#fff3cd', 'color': '#856404'},
                            ]
                        )
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader("Job Logs"),
                    dbc.CardBody([
                        html.Div(id="selected-job-info", className="mb-2 fw-bold", children="No job selected"),
                        html.Pre(
                            id="job-log-display",
                            className="p-3 bg-dark text-light rounded small",
                            style={"height": "300px", "overflowY": "auto", "fontFamily": "monospace", "whiteSpace": "pre-wrap"},
                            children="Select a job from the table to view its execution logs."
                        )
                    ])
                ], className="mt-3")
            ], md=8)
        ])
    ], fluid=True)
