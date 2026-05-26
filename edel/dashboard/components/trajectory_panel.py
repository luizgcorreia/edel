"""Trajectory Explorer Panel Component for the EDEL Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from edel.dashboard.utils import get_registry_options


def trajectory_panel_layout() -> dbc.Container:
    """Layout for the Trajectory Explorer panel (Tab 6)."""
    return dbc.Container([
        html.H3("Trajectory Explorer", className="mt-3 mb-1"),
        html.P(
            "Analyse the epistemic trajectory of a paper (by Work ID) or explore a synthetic "
            "hypothesis (by entering custom text segments). Neighbours are found at each "
            "epistemic stop: Problem → Method → Finding → Interpretation.",
            className="text-muted mb-4"
        ),

        dbc.Row([
            # ----------------------------------------------------------------
            # Left column — Controls
            # ----------------------------------------------------------------
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Configuration"),
                    dbc.CardBody([

                        # Experiment
                        html.Label("Experiment:", className="fw-bold"),
                        dcc.Dropdown(
                            id="traj-experiment-select",
                            options=get_registry_options(),
                            clearable=False,
                            persistence=False,
                            className="mb-3"
                        ),

                        # Projection Method
                        html.Label("Projection Method:", className="fw-bold"),
                        dcc.RadioItems(
                            id="traj-method-select",
                            options=[
                                {"label": " Diffusion", "value": "diffusion"},
                                {"label": " UMAP", "value": "umap"},
                            ],
                            value="diffusion",
                            inline=True,
                            className="mb-3"
                        ),

                        # Distance Space
                        html.Label("Distance Space:", className="fw-bold"),
                        dcc.RadioItems(
                            id="traj-space-select",
                            options=[
                                {"label": " Embedding (cosine)", "value": "embedding"},
                                {"label": " 2D Projection (Euclidean)", "value": "2d"},
                            ],
                            value="embedding",
                            inline=False,
                            className="mb-3"
                        ),

                        html.Hr(),

                        # Search mode toggle
                        html.Label("Input Mode:", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button("By Work ID", id="traj-mode-id-btn",
                                       color="primary", size="sm", outline=False),
                            dbc.Button("Synthetic Text", id="traj-mode-synthetic-btn",
                                       color="primary", size="sm", outline=True),
                        ], className="mb-3 w-100"),
                        dcc.Store(id="traj-input-mode", data="work_id"),

                        # Work ID input
                        html.Div(id="traj-workid-input-group", children=[
                            html.Label("Work ID:"),
                            dbc.Input(
                                id="traj-paper-id",
                                placeholder="https://openalex.org/W...",
                                type="text",
                                className="mb-2"
                            ),
                            html.Small(
                                "The paper must exist in the selected experiment's dataset.",
                                className="text-muted"
                            )
                        ], className="mb-3"),

                        # Synthetic text inputs (hidden by default)
                        html.Div(id="traj-synthetic-input-group", children=[
                            html.Label("Problem Segment:"),
                            dbc.Textarea(
                                id="traj-segment-problem",
                                placeholder="Describe the problem addressed...",
                                rows=3, className="mb-2"
                            ),
                            html.Label("Method Segment:"),
                            dbc.Textarea(
                                id="traj-segment-method",
                                placeholder="Describe the methodology...",
                                rows=3, className="mb-2"
                            ),
                            html.Label("Finding Segment:"),
                            dbc.Textarea(
                                id="traj-segment-finding",
                                placeholder="Describe the main finding...",
                                rows=3, className="mb-2"
                            ),
                            html.Label("Interpretation Segment:"),
                            dbc.Textarea(
                                id="traj-segment-interpretation",
                                placeholder="Describe the interpretation/implications...",
                                rows=3, className="mb-2"
                            ),
                        ], style={"display": "none"}, className="mb-3"),

                        html.Hr(),

                        # Neighbour parameters
                        html.Label("Neighbour Parameters:", className="fw-bold"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("K (nearest):", className="small text-muted"),
                                dbc.Input(
                                    id="traj-k",
                                    type="number",
                                    value=5,
                                    min=1,
                                    step=1,
                                    className="mb-2"
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Radius (optional):", className="small text-muted"),
                                dbc.Input(
                                    id="traj-radius",
                                    type="number",
                                    placeholder="e.g. 0.2",
                                    min=0,
                                    step=0.01,
                                    className="mb-2"
                                ),
                                html.Small("Overrides K when set.", className="text-muted"),
                            ], width=6),
                        ]),

                        html.Hr(),

                        # Run button
                        dcc.Loading(
                            id="traj-loading",
                            type="circle",
                            children=[
                                dbc.Button(
                                    "Run Trajectory Analysis",
                                    id="traj-run-btn",
                                    color="primary",
                                    className="w-100 mb-2",
                                    disabled=True
                                ),
                            ]
                        ),
                        dbc.Button(
                            "Download Report (.md)",
                            id="traj-download-btn",
                            color="secondary",
                            outline=True,
                            className="w-100",
                            style={"display": "none"}  # Hidden until analysis is run
                        ),
                        dcc.Download(id="traj-download-component"),
                        html.Div(id="traj-status-msg", className="mt-2 small text-danger"),
                    ])
                ])
            ], md=3),

            # ----------------------------------------------------------------
            # Right column — Results
            # ----------------------------------------------------------------
            dbc.Col([

                # Plots Row (2D and 3D)
                dcc.Store(id="traj-selected-vertex", data="problem"),
                dbc.Row([
                    # 2D Trajectory Plot
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Span("2D Epistemic Displacement (P Space)"),
                                dbc.Button("⛶ Fullscreen", id="traj-plot-fullscreen-btn", size="sm", color="link", className="float-end p-0 text-decoration-none text-info")
                            ]),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="traj-plot",
                                    figure={"data": [], "layout": {
                                        "template": "plotly_dark",
                                        "paper_bgcolor": "#1a1a2e",
                                        "plot_bgcolor": "#16213e",
                                        "height": 380,
                                        "annotations": [{
                                            "text": "Run an analysis to see the 2D displacement.",
                                            "xref": "paper", "yref": "paper",
                                            "x": 0.5, "y": 0.5,
                                            "showarrow": False,
                                            "font": {"color": "#aaa", "size": 14}
                                        }]
                                    }},
                                    style={"height": "380px"},
                                    config={"displayModeBar": False}
                                ),
                                html.Small(
                                    "Note: 2D projection is only available for Work ID inputs where coordinates exist.",
                                    className="text-muted mt-1"
                                )
                            ])
                        ], id="traj-plot-2d-card", className="mb-3")
                    ], md=6),

                    # 3D Tetrahedron Plot
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Span("3D Epistemic Simplex (Tetrahedron)"),
                                dbc.Button("⛶ Fullscreen", id="traj-plot-3d-fullscreen-btn", size="sm", color="link", className="float-end p-0 text-decoration-none text-info")
                            ]),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="traj-plot-3d",
                                    figure={"data": [], "layout": {
                                        "template": "plotly_dark",
                                        "paper_bgcolor": "#1a1a2e",
                                        "plot_bgcolor": "#16213e",
                                        "height": 380,
                                        "annotations": [{
                                            "text": "Run an analysis to see the 3D Simplex.",
                                            "xref": "paper", "yref": "paper",
                                            "x": 0.5, "y": 0.5,
                                            "showarrow": False,
                                            "font": {"color": "#aaa", "size": 14}
                                        }]
                                    }},
                                    style={"height": "380px"},
                                    config={"displayModeBar": True}
                                ),
                                html.Small(
                                    "Tip: Click on a vertex (P, M, F, I) to inspect its discourse space & neighbors.",
                                    className="text-muted mt-1"
                                )
                            ])
                        ], id="traj-plot-3d-card", className="mb-3")
                    ], md=6)
                ], className="mb-3"),

                # Selected Vertex Segment Text Display
                dbc.Card([
                    dbc.CardHeader([
                        "Active Simplex Space: ",
                        html.Span("Problem", id="traj-selected-vertex-label", className="badge bg-warning ms-1 text-dark")
                    ]),
                    dbc.CardBody([
                        html.P(
                            id="traj-selected-segment-text",
                            className="mb-0 text-white font-monospace small",
                            style={"whiteSpace": "pre-wrap"}
                        )
                    ])
                ], id="traj-selected-vertex-container", className="mb-3", style={"display": "none"}),

                # Trajectory & Operator Metrics
                dbc.Card([
                    dbc.CardHeader("Trajectory & Operator Metrics"),
                    dbc.CardBody([
                        html.Div(
                            id="traj-metrics-container",
                            children=[
                                html.P(
                                    "Metrics will appear here after running the analysis.",
                                    className="text-muted mb-0"
                                )
                            ]
                        )
                    ])
                ], className="mb-3"),

                # Results accordion
                dbc.Card([
                    dbc.CardHeader("Neighbourhood Results"),
                    dbc.CardBody([
                        html.Div(
                            id="traj-results-container",
                            children=[
                                html.P(
                                    "Results will appear here after running the analysis.",
                                    className="text-muted"
                                )
                            ]
                        )
                    ])
                ])
            ], md=9)
        ])
    ], fluid=True, className="mb-5")
