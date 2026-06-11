"""Report Generator Panel Component for the EDEL Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def report_generator_panel_layout() -> dbc.Container:
    """Layout for the Report Generator panel."""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H3("Report Generator"), width=12),
        ], className="mt-3 mb-4"),

        dbc.Row([
            # Left Column: Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Report Configuration"),
                    dbc.CardBody([
                        html.Label("Experiments:", className="fw-bold"),
                        dcc.Dropdown(
                            id="report-experiment-select",
                            options=[],
                            multi=True,
                            placeholder="Select experiments...",
                            className="mb-3"
                        ),
                        dbc.Button(
                            "Select All / None",
                            id="report-toggle-all-btn",
                            color="secondary",
                            outline=True,
                            size="sm",
                            className="mb-3"
                        ),

                        html.Label("Control / Null Run (optional):", className="fw-bold"),
                        dcc.Dropdown(
                            id="report-control-select",
                            options=[],
                            multi=False,
                            placeholder="Select control for H1b...",
                            className="mb-3"
                        ),

                        html.Label("Hypotheses:", className="fw-bold"),
                        dcc.Checklist(
                            id="report-hypothesis-checklist",
                            options=[
                                {"label": " H1 – Structural Transition", "value": "H1"},
                                {"label": " H2 – Local Transition Organization", "value": "H2"},
                                {"label": " H3 – Predictive Capacity", "value": "H3"},
                            ],
                            value=["H1", "H2", "H3"],
                            inputStyle={"margin-right": "8px"},
                            className="mb-3"
                        ),

                        html.Label("Mode:", className="fw-bold"),
                        dcc.RadioItems(
                            id="report-mode-radio",
                            options=[
                                {"label": " Use cached results (fast)", "value": "cache"},
                                {"label": " Recompute from artifacts (--force)", "value": "force"},
                            ],
                            value="cache",
                            inputStyle={"margin-right": "8px"},
                            className="mb-3"
                        ),

                        html.Hr(className="my-3"),
                        html.H6("Subsample Overrides (force mode only)", className="fw-bold mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("H1 N:", className="small"),
                                dcc.Input(
                                    id="report-h1-n",
                                    type="number",
                                    min=100, max=5000, step=100,
                                    value=500,
                                    className="form-control form-control-sm mb-2"
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("H3 N:", className="small"),
                                dcc.Input(
                                    id="report-h3-n",
                                    type="number",
                                    min=500, max=10000, step=500,
                                    value=2000,
                                    className="form-control form-control-sm mb-2"
                                ),
                            ], width=6),
                        ]),

                        dbc.Button(
                            " Generate Report (.xlsx)",
                            id="btn-generate-report",
                            color="primary",
                            className="w-100 mt-2"
                        ),

                        html.Div(id="report-status-msg", className="mt-3 small"),
                    ])
                ])
            ], md=4),

            # Right Column: Preview / Instructions
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Report Preview"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="report-loading",
                            type="circle",
                            children=html.Div(
                                id="report-preview-container",
                                children=[
                                    html.Div(
                                        "Select experiments and hypotheses, then click 'Generate Report' to produce an Excel workbook. "
                                        "Each hypothesis gets its own tab with metadata, primary metrics, and diagnostic columns.",
                                        className="text-center text-muted p-5 my-4 border rounded bg-light"
                                    )
                                ]
                            )
                        )
                    ])
                ])
            ], md=8)
        ]),

        dcc.Download(id="report-download-component"),
        dcc.Store(id="report-store"),
    ], fluid=True)
