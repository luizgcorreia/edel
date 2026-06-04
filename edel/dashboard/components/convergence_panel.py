"""Convergence Analysis Panel Component for the EDEL Dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def convergence_panel_layout() -> dbc.Container:
    """Layout for the Convergence Analysis and Calibration panel."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Convergence Analysis & Sample Calibration", className="text-primary"),
                html.P(
                    "Calibrate dataset sizes by checking how sample statistics for H1, H2, and H3 "
                    "converge to the full-dataset 'ground truth' values.",
                    className="lead text-muted"
                )
            ], width=12),
        ], className="mt-3 mb-4"),

        dbc.Row([
            # Left Column: Configuration Selector
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Experiment", className="fw-bold bg-light"),
                    dbc.CardBody([
                        html.Label("Target Experiment Run:", className="fw-bold small"),
                        dcc.Dropdown(
                            id="conv-experiment-select",
                            options=[],
                            placeholder="Select experiment...",
                            className="mb-3"
                        ),
                        
                        dbc.Button(
                            "⚡ Run Convergence Analysis",
                            id="btn-run-conv-analysis",
                            color="primary",
                            className="w-100 mt-2"
                        )
                    ])
                ], className="mb-4 shadow-sm"),
                
                dbc.Card([
                    dbc.CardHeader("Calibration Objectives", className="fw-bold bg-light"),
                    dbc.CardBody([
                        html.Div([
                            html.H6("H1 Convergence", className="text-secondary fw-bold mb-1"),
                            html.P(
                                "Determine the sample size n at which the structural transition "
                                "statistics (KS statistic, 1D Wasserstein) become indistinguishable "
                                "from the full dataset.",
                                className="small text-muted mb-3"
                            ),
                            
                            html.H6("H2 Convergence", className="text-secondary fw-bold mb-1"),
                            html.P(
                                "Trace the Mean Absolute Error (MAE) of transition z-scores across all 12 "
                                "transitions, finding where local organizational patterns stabilize.",
                                className="small text-muted mb-3"
                            ),
                            
                            html.H6("H3 Calibration", className="text-secondary fw-bold mb-1"),
                            html.P(
                                "Compare Uniform vs Temporal Stratified sampling at various fractions "
                                "(5% to 50%) to identify the most cost-effective subset for forecasting.",
                                className="small text-muted mb-0"
                            )
                        ])
                    ])
                ], className="shadow-sm")
            ], md=4),
            
            # Right Column: Results & Report
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Convergence Study & Report", className="fw-bold bg-light"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="conv-loading",
                            type="circle",
                            children=html.Div(
                                id="conv-report-container",
                                children=[
                                    html.Div(
                                        "Select a run from the dropdown and click 'Run Convergence Analysis'. "
                                        "Cached results will load instantly.",
                                        className="text-center text-muted p-5 my-4 border rounded bg-light"
                                    )
                                ]
                            )
                        )
                    ])
                ], className="shadow-sm mb-4")
            ], md=8)
        ])
    ], fluid=True)
