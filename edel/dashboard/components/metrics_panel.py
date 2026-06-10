"""Metrics and Analysis Component for the EDEL Dashboard."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

def metrics_panel_layout() -> dbc.Container:
    """Layout for the Metrics Analysis panel."""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H3("Metrics Analysis"), width=8),
            dbc.Col(
                dbc.Button("↻ Rebuild Cache", id="btn-rebuild-cache", color="warning", className="float-end"),
                width=4
            )
        ], className="mt-3 mb-4"),
        
        dbc.Row([
            # Left Column: Filters & KS Tests
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters"),
                    dbc.CardBody([
                        html.Label("Metric to Plot:"),
                        dcc.Dropdown(
                            id="metric-y-axis",
                            options=[
                                {"label": "Segmentation Ratio", "value": "seg_ratio_mean"},
                                {"label": "Silhouette (Transitions)", "value": "silhouette_transitions"},
                                {"label": "Silhouette (Features)", "value": "silhouette_features"},
                                {"label": "Cosine (P-M)", "value": "sim_pm"},
                                {"label": "Norm (P-M)", "value": "norm_pm"}
                            ],
                            value="silhouette_transitions",
                            clearable=False,
                            className="mb-3"
                        ),
                        html.Hr(),
                        html.Label("KS Tests:"),
                        dcc.Dropdown(
                            id="ks-feature-select",
                            options=[
                                {"label": "Norm P-M", "value": "norm_pm_dist"},
                                {"label": "Norm M-F", "value": "norm_mf_dist"},
                                {"label": "Norm F-I", "value": "norm_fi_dist"},
                                {"label": "Norm P-F", "value": "norm_pf_dist"},
                                {"label": "Norm P-I", "value": "norm_pi_dist"},
                                {"label": "Norm M-I", "value": "norm_mi_dist"},
                                {"label": "Cosine (PM, MF)", "value": "cos_pm_mf_dist"},
                                {"label": "Cosine (PM, FI)", "value": "cos_pm_fi_dist"},
                                {"label": "Cosine (MF, FI)", "value": "cos_mf_fi_dist"},
                            ],
                            value="cos_pm_mf_dist",
                            clearable=False,
                            className="mb-2"
                        ),
                        dbc.Button("Run KS Tests", id="btn-run-ks", color="info", className="w-100")
                    ])
                ])
            ], md=3),
            
            # Right Column: Data & Plots
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Experiment Dataset"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='metrics-table',
                            columns=[],
                            data=[],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '5px'},
                            sort_action="native",
                            filter_action="native"
                        )
                    ])
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Comparison Plot"),
                            dbc.CardBody([
                                dcc.Graph(id='metrics-bar-chart')
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("KS Test Results (p-values)"),
                            dbc.CardBody([
                                dcc.Graph(id='ks-heatmap-chart')
                            ])
                        ])
                    ], md=6)
                ])
            ], md=9)
        ])
    ], fluid=True)
