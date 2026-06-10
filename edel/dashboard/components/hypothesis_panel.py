"""Hypothesis Testing Panel Component for the EDEL Dashboard."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from edel.dashboard.utils import get_registry_options

def hypothesis_panel_layout() -> dbc.Container:
    """Layout for the Hypothesis Testing and Comparison panel."""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H3("Hypothesis Testing & Validation"), width=12),
        ], className="mt-3 mb-4"),

        dbc.Row([
            # Left Column: Configuration Selector
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Configure Hypothesis Experiment"),
                    dbc.CardBody([
                        html.Label("Hypothesis / Observed Run:", className="fw-bold"),
                        dcc.Dropdown(
                            id="hyp-hypothesis-select",
                            options=[],
                            placeholder="Select observed run...",
                            className="mb-3"
                        ),
                        
                        html.Label("Control / Null Run:", className="fw-bold"),
                        dcc.Dropdown(
                            id="hyp-control-select",
                            options=[],
                            placeholder="Select control/null run...",
                            className="mb-3"
                        ),
                        
                        dbc.Button(
                            "⚡ Run Hypothesis Tests",
                            id="btn-run-hyp-tests",
                            color="primary",
                            className="w-100 mt-2"
                        )
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Hypothesis Overview"),
                    dbc.CardBody([
                        html.Div([
                            html.H6("H1: Structural Transition", className="text-primary"),
                            html.P("Tests if epistemic trajectories have structured coupling via multivariate energy distance on 6D transition features (3 sequential norms + 3 cosines). Per-edge Wasserstein effect sizes and KS diagnostics provided as secondary.", className="small text-muted"),
                            
                            html.H6("H2: Local Transition Organization", className="text-primary mt-3"),
                            html.P("Tests if local neighborhood transitions are statistically constrained (Wasserstein permutation test). Asymmetry metrics characterize directionality bias as secondary.", className="small text-muted"),
                            
                            html.H6("H3: Predictive Transition Capacity", className="text-primary mt-3"),
                            html.P("Tests if transition operators forecast future problem spaces better than a persistence baseline, and checks spatial alignment using Bivariate Moran's I.", className="small text-muted")
                        ])
                    ])
                ])
            ], md=4),
            
            # Right Column: Results & Report
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Experiment Validation Report"),
                    dbc.CardBody([
                        # Status message / spinner while running
                        dcc.Loading(
                            id="hyp-loading",
                            type="circle",
                            children=html.Div(
                                id="hyp-report-container",
                                children=[
                                    html.Div(
                                        "Select a Hypothesis run and a Control run, then click 'Run Hypothesis Tests' to generate the validation report.",
                                        className="text-center text-muted p-5 my-4 border rounded bg-light"
                                    )
                                ]
                            )
                        )
                    ])
                ])
            ], md=8)
        ])
    ], fluid=True)
