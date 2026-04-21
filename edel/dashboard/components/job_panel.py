"""Experiment Runner Component for the EDEL Dashboard."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from edel.dashboard.utils import get_registry_options

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
                            value="scientometrics_baseline",
                            clearable=False,
                            className="mb-3"
                        ),
                        
                        html.H6("Sweep Axes (Optional)", className="mt-3"),
                        html.Div(id="sweep-axes-container", children=[
                            dbc.Row([
                                dbc.Col(dbc.Label("Embedding Model"), width=4),
                                dbc.Col(dbc.Checklist(
                                    options=[
                                        {"label": "ada-002", "value": "text-embedding-ada-002"},
                                        {"label": "3-small", "value": "text-embedding-3-small"}
                                    ],
                                    value=["text-embedding-ada-002"],
                                    id="sweep-embedding",
                                    inline=True,
                                    switch=True,
                                ), width=8)
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Projection"), width=4),
                                dbc.Col(dbc.Checklist(
                                    options=[
                                        {"label": "diffusion", "value": "diffusion"},
                                        {"label": "umap", "value": "umap"}
                                    ],
                                    value=["diffusion"],
                                    id="sweep-projection",
                                    inline=True,
                                    switch=True,
                                ), width=8)
                            ], className="mb-2"),
                        ]),
                        
                        html.Hr(),
                        html.Div(id="sweep-summary", className="mb-3 font-weight-bold"),
                        dbc.Button("Submit Jobs", id="btn-submit-jobs", color="success", className="w-100")
                    ])
                ])
            ], md=4),
            
            # Right Column: Job Queue
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col("Job Queue"),
                            dbc.Col(
                                dbc.Button("Refresh", id="btn-refresh-jobs", size="sm", color="secondary", outline=True),
                                className="text-end"
                            )
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
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{status} = "done"'}, 'backgroundColor': '#d4edda', 'color': '#155724'},
                                {'if': {'filter_query': '{status} = "failed"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
                                {'if': {'filter_query': '{status} = "running"'}, 'backgroundColor': '#fff3cd', 'color': '#856404'},
                            ]
                        )
                    ])
                ])
            ], md=8)
        ])
    ], fluid=True)
