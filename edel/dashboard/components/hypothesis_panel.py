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
                            html.H6("H1a: Structural Transition", className="text-primary"),
                            html.P("Passes if multivariate energy distance p < 0.05 (pooled permutation test on 6D features: 3 sequential norms + 3 cosines vs within-paper shuffled null). Per-edge Wasserstein effect sizes and KS diagnostics reported as secondary.", className="small text-muted"),

                            html.H6("H1b: Scientific Specificity", className="text-primary mt-3"),
                            html.P("Passes if energy distance p < 0.05 comparing the 6D feature distribution against the selected control experiment. Experiments with \"null\" in the name always return p = 1.0 (identity test).", className="small text-muted"),

                            html.H6("H2: Local Transition Organization", className="text-primary mt-3"),
                            html.P("Passes if at least 6 of 12 directional aspect-to-aspect transitions have p < 0.05 (Wasserstein distance permutation test). Asymmetry metrics (H2b) characterize directionality bias as secondary.", className="small text-muted"),

                            html.H6("H3: Predictive Transition Capacity", className="text-primary mt-3"),
                            html.P("Passes if predictive gain > 0 AND temporal permutation p < 0.05 (Ridge transition operator forecast vs persistence baseline). Spatial alignment assessed via Bivariate Moran's I.", className="small text-muted")
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
