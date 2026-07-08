"""Robustness Tests Panel Component for the EDEL Dashboard."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

def robustness_panel_layout() -> dbc.Container:
    """Layout for the Robustness Tests panel."""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H3("Null Model Robustness Tests"), width=12),
        ], className="mt-3 mb-4"),

        dbc.Row([
            # Left Column: Configuration Selector
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Experiment Selection"),
                    dbc.CardBody([
                        html.Label("Primary Experiment:", className="fw-bold"),
                        dcc.Dropdown(
                            id="rob-experiment-select",
                            options=[],
                            placeholder="Select primary experiment...",
                            className="mb-3"
                        ),
                        
                        html.Label("Comparison Experiment (optional):", className="fw-bold"),
                        dcc.Dropdown(
                            id="rob-compare-experiment-select",
                            options=[],
                            placeholder="Select experiment for cross-model comparison...",
                            className="mb-3"
                        ),
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Document Sampling"),
                    dbc.CardBody([
                        html.Label("Search & Select Papers (manual):", className="fw-bold"),
                        dcc.Dropdown(
                            id="rob-doc-search",
                            placeholder="Type to search and select papers...",
                            searchable=True,
                            multi=True,
                            options=[],
                            className="mb-3"
                        ),
                        
                        html.Label("Sample Size:", className="fw-bold"),
                        dbc.InputGroup([
                            dbc.Input(id="rob-sample-size", type="number", value=10, min=1),
                            dbc.Button("Sample", id="btn-rob-sample", color="secondary"),
                        ], className="mb-3"),
                        
                        html.Div(id="rob-selected-count", className="text-muted small"),
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Test Configuration"),
                    dbc.CardBody([
                        html.Label("Tests to Run:", className="fw-bold"),
                        dcc.Checklist(
                            id="rob-test-select",
                            options=[],  # Populated via callback
                            value=[],
                            className="mb-3 checklist-custom",
                            inputClassName="me-2",
                            labelClassName="mb-1 d-block"
                        ),
                        
                        html.Label("Perturbation Intensity (N):", className="fw-bold mt-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Max N", className="small text-muted"),
                                dbc.Input(id="rob-n-max", type="number", value=30, min=1)
                            ]),
                            dbc.Col([
                                html.Label("Step", className="small text-muted"),
                                dbc.Input(id="rob-n-step", type="number", value=1, min=1)
                            ])
                        ], className="mb-4"),
                        
                        dbc.Button(
                            "⚡ Run Robustness Tests",
                            id="btn-run-robustness",
                            color="primary",
                            className="w-100"
                        )
                    ])
                ], className="mb-4"),
            ], md=4),
            
            # Right Column: Results
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Robustness Sweep Results"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(
                                    options=[
                                        {"label": "Show ± 1 Std Dev Band", "value": "std"},
                                        {"label": "Show Individual Traces", "value": "traces"}
                                    ],
                                    value=[],
                                    id="rob-plot-options",
                                    inline=True,
                                    switch=True,
                                )
                            ], className="mb-3")
                        ]),
                        
                        dcc.Loading(
                            id="rob-loading",
                            type="circle",
                            children=html.Div(
                                id="rob-report-container",
                                children=[
                                    html.Div(
                                        "Select experiments, sample documents, and tests to run, then click 'Run Robustness Tests'.",
                                        className="text-center text-muted p-5 my-4 border rounded bg-light"
                                    )
                                ]
                            )
                        )
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Structural Correlations (Static)"),
                    dbc.CardBody([
                        html.P(
                            "This plots the relationship between text length differences and embedding displacement "
                            "across aspects (e.g., Method vs Problem) for the entire selected experiment.",
                            className="text-muted small"
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Aspect 1 (X_1)", className="small fw-bold"),
                                dbc.Select(
                                    id="rob-struct-aspect1",
                                    options=[
                                        {"label": "Problem", "value": "problem"},
                                        {"label": "Method", "value": "method"},
                                        {"label": "Finding", "value": "finding"},
                                        {"label": "Interpretation", "value": "interpretation"}
                                    ],
                                    value="method"
                                )
                            ]),
                            dbc.Col([
                                html.Label("Aspect 2 (X_2)", className="small fw-bold"),
                                dbc.Select(
                                    id="rob-struct-aspect2",
                                    options=[
                                        {"label": "Problem", "value": "problem"},
                                        {"label": "Method", "value": "method"},
                                        {"label": "Finding", "value": "finding"},
                                        {"label": "Interpretation", "value": "interpretation"}
                                    ],
                                    value="problem"
                                )
                            ])
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Correlation Metric", className="small fw-bold"),
                                dbc.Select(
                                    id="rob-struct-metric",
                                    options=[
                                        {"label": "Word Length Difference", "value": "length"},
                                        {"label": "Sentence Count Difference", "value": "sentence_count"},
                                        {"label": "PMFI Length Ratio Difference", "value": "pmfi_ratio"},
                                        {"label": "Descriptive-to-Noun Ratio Difference", "value": "descriptive_noun_ratio"}
                                    ],
                                    value="length"
                                )
                            ])
                        ], className="mb-3"),
                        dbc.Button("Generate Scatter Plot", id="btn-run-structural", color="info", size="sm", className="mb-3"),
                        
                        dcc.Loading(
                            type="circle",
                            children=html.Div(id="rob-structural-container")
                        )
                    ])
                ])
            ], md=8)
        ])
    ], fluid=True)
