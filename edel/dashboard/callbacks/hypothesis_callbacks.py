"""Callbacks for the Hypothesis Testing and Validation panel."""

import pickle
import logging
from pathlib import Path
from dash import Dash, Input, Output, State, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from edel.experiments.runner import load_registry
from edel.dashboard.cache import get_results_df

logger = logging.getLogger(__name__)

def _load_features(exp_id: str, base_path: Path) -> dict | None:
    """Load cached features (distributions) for a given experiment ID."""
    features_path = base_path / "experiments" / exp_id / "features.pkl"
    if not features_path.exists():
        return None
    try:
        with open(features_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Error loading features for '{exp_id}': {e}")
        return None

def register_hypothesis_callbacks(app: Dash, base_path: Path) -> None:
    
    # Callback to populate the run selection dropdowns from registry
    @app.callback(
        [Output("hyp-hypothesis-select", "options"),
         Output("hyp-control-select", "options")],
        [Input("artifact-update-store", "data")]
    )
    def update_run_selectors(update_val):
        try:
            registry = load_registry(base_path)
            options = [{"label": exp["experiment_id"], "value": exp["experiment_id"]} for exp in registry]
            return options, options
        except Exception as e:
            logger.error(f"Error updating run selectors: {e}")
            return [], []

    # Callback to run the hypothesis tests and generate the report
    @app.callback(
        Output("hyp-report-container", "children"),
        [Input("btn-run-hyp-tests", "n_clicks")],
        [State("hyp-hypothesis-select", "value"),
         State("hyp-control-select", "value")],
        prevent_initial_call=True
    )
    def generate_hypothesis_report(n_clicks, hyp_id, ctrl_id):
        if not n_clicks:
            return html.Div("Select a Hypothesis run and a Control run, then click 'Run Hypothesis Tests' to generate the validation report.", className="text-center text-muted p-5 my-4 border rounded bg-light")
            
        if not hyp_id or not ctrl_id:
            return dbc.Alert("Please select both a Hypothesis run and a Control run.", color="warning", className="mt-3")

        try:
            # Load results dataframe (which contains individual run metrics)
            df = get_results_df(base_path)
            if df.empty:
                return dbc.Alert("No analyzed results found. Please rebuild the results cache on the 'Metrics Analysis' tab.", color="danger", className="mt-3")

            hyp_rows = df[df["experiment_id"] == hyp_id]
            ctrl_rows = df[df["experiment_id"] == ctrl_id]

            if hyp_rows.empty or ctrl_rows.empty:
                return dbc.Alert("One or both selected runs have not been analyzed yet. Please rebuild the cache on the 'Metrics Analysis' tab first.", color="danger", className="mt-3")

            hyp_metrics = hyp_rows.iloc[0].to_dict()
            ctrl_metrics = ctrl_rows.iloc[0].to_dict()

            # Load features for direct comparison
            feat_hyp = _load_features(hyp_id, base_path)
            feat_ctrl = _load_features(ctrl_id, base_path)

            # 1. Compare feature distributions directly (H1)
            direct_h1_results = {}
            if feat_hyp and feat_ctrl:
                dims = [
                    ("norm_pm_dist", "Norm P-M"),
                    ("norm_mf_dist", "Norm M-F"),
                    ("norm_fi_dist", "Norm F-I"),
                    ("cos_pm_mf_dist", "Cosine (PM, MF)"),
                    ("cos_pm_fi_dist", "Cosine (PM, FI)"),
                    ("cos_mf_fi_dist", "Cosine (MF, FI)"),
                ]
                for key, label in dims:
                    if key in feat_hyp and key in feat_ctrl:
                        res = ks_2samp(feat_hyp[key], feat_ctrl[key])
                        direct_h1_results[label] = {
                            "stat": float(res.statistic),
                            "pvalue": float(res.pvalue)
                        }

            # Determine if H1, H2, H3 are supported
            # H1: Supported if KS test p-values are small for the Hypothesis run
            h1_pvals = [hyp_metrics.get(f"h1_ks_pvalue_{k}", 1.0) for k in ["norm_pm", "norm_mf", "norm_fi", "cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]]
            h1_supported = sum(p < 0.05 for p in h1_pvals) >= 3

            # H2: Supported if local neighborhoods show significant clustering
            h2_pvals = [hyp_metrics.get(f"h2_pvalue_{k}", 1.0) for k in ["pm", "mf", "fi"]]
            h2_supported = sum(p < 0.05 for p in h2_pvals) >= 2

            # H3: Supported if predictive gain is positive and Moran's I is significant
            h3_supported = (hyp_metrics.get("h3_moran_pvalue", 1.0) < 0.05) and (hyp_metrics.get("h3_predictive_gain", -1) > 0)

            report_children = []

            def make_badge(supported: bool):
                if supported:
                    return dbc.Badge("SUPPORTED", color="success", className="px-2 py-1")
                else:
                    return dbc.Badge("NOT SUPPORTED", color="danger", className="px-2 py-1")

            # Validation Dashboard Summary Card
            report_children.append(html.Div([
                html.H4("Validation Summary Dashboard", className="border-bottom pb-2 mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("H1: Structural Shift", className="card-title"),
                            html.Div([make_badge(h1_supported)]),
                            html.P("Trajectories differ significantly from random aspect-shuffling.", className="small text-muted mt-2 mb-0")
                        ])
                    ], color="success" if h1_supported else "light", outline=True), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("H2: Local Clustering", className="card-title"),
                            html.Div([make_badge(h2_supported)]),
                            html.P("Transitions show neighborhood-level spatial organization.", className="small text-muted mt-2 mb-0")
                        ])
                    ], color="success" if h2_supported else "light", outline=True), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H5("H3: Predictive Power", className="card-title"),
                            html.Div([make_badge(h3_supported)]),
                            html.P("Historical model predicts future locations successfully.", className="small text-muted mt-2 mb-0")
                        ])
                    ], color="success" if h3_supported else "light", outline=True), md=4),
                ], className="mb-4")
            ]))

            # H1 Details Section
            h1_rows = []
            for k in ["norm_pm", "norm_mf", "norm_fi", "cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]:
                h1_rows.append(html.Tr([
                    html.Td(k.upper().replace("_", " ")),
                    html.Td(f"{hyp_metrics.get(f'h1_ks_stat_{k}', 0.0):.4f}"),
                    html.Td(f"{hyp_metrics.get(f'h1_ks_pvalue_{k}', 1.0):.4g}"),
                    html.Td(f"{ctrl_metrics.get(f'h1_ks_stat_{k}', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get(f'h1_ks_pvalue_{k}', 1.0):.4g}"),
                ]))
                
            h1_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Feature Dimension"),
                    html.Th("Hyp KS Stat"),
                    html.Th("Hyp p-val"),
                    html.Th("Ctrl KS Stat"),
                    html.Th("Ctrl p-val"),
                ])),
                html.Tbody(h1_rows)
            ], className="table table-striped table-sm small border")

            # Direct H1 Comparison
            direct_h1_rows = []
            for label, val in direct_h1_results.items():
                direct_h1_rows.append(html.Tr([
                    html.Td(label),
                    html.Td(f"{val['stat']:.4f}"),
                    html.Td(f"{val['pvalue']:.4g}"),
                ]))
            direct_h1_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Feature Dimension"),
                    html.Th("KS Statistic (Hyp vs Ctrl)"),
                    html.Th("p-value"),
                ])),
                html.Tbody(direct_h1_rows)
            ], className="table table-bordered table-sm small border mt-3")

            report_children.append(html.Div([
                html.H5("H1: Structural Transition Details", className="mt-4 text-primary"),
                html.P("Hypothesis 1 asserts that epistemic trajectories have a structured coupling. Shuffling aspects breaks this coupling. Lower p-values (< 0.05) show structured transition behaviors.", className="small text-muted"),
                h1_table,
                html.H6("Direct Comparison (Hypothesis vs. Control)"),
                html.P("Performs a two-sample Kolmogorov-Smirnov test directly comparing the trajectory distributions of both runs.", className="small text-muted"),
                direct_h1_table if direct_h1_results else html.P("No feature distributions available for direct comparison.", className="text-muted small")
            ]))

            # H2 Details Section
            h2_rows = []
            for k in ["pm", "mf", "fi"]:
                h2_rows.append(html.Tr([
                    html.Td(f"{k.upper()} Transition"),
                    html.Td(f"{hyp_metrics.get(f'h2_w_dist_{k}', 0.0):.4f}"),
                    html.Td(f"{hyp_metrics.get(f'h2_pvalue_{k}', 1.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get(f'h2_w_dist_{k}', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get(f'h2_pvalue_{k}', 1.0):.4f}"),
                ]))
            h2_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Transition Space"),
                    html.Th("Hyp Wasserstein (Obs vs Rand)"),
                    html.Th("Hyp Perm p-val"),
                    html.Th("Ctrl Wasserstein (Obs vs Rand)"),
                    html.Th("Ctrl Perm p-val"),
                ])),
                html.Tbody(h2_rows)
            ], className="table table-striped table-sm small border")

            report_children.append(html.Div([
                html.H5("H2: Local Transition Organization Details", className="mt-4 text-primary"),
                html.P("Hypothesis 2 evaluates if local neighborhood transitions are statistically constrained/clustered. A significant permuted p-value (< 0.05) supports localized organizational constraints.", className="small text-muted"),
                h2_table
            ]))

            # H3 Details Section
            h3_rows = [
                html.Tr([
                    html.Td("Global W_EDEL (Wasserstein prediction error)"),
                    html.Td(f"{hyp_metrics.get('h3_w_edel', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h3_w_edel', 0.0):.4f}"),
                ]),
                html.Tr([
                    html.Td("Global W_baseline (persistence prediction error)"),
                    html.Td(f"{hyp_metrics.get('h3_w_baseline', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h3_w_baseline', 0.0):.4f}"),
                ]),
                html.Tr([
                    html.Td("Predictive Gain (W_baseline - W_EDEL)"),
                    html.Td(f"{hyp_metrics.get('h3_predictive_gain', 0.0):.4f}", className="text-success fw-bold" if hyp_metrics.get('h3_predictive_gain', 0) > 0 else "text-danger"),
                    html.Td(f"{ctrl_metrics.get('h3_predictive_gain', 0.0):.4f}", className="text-success fw-bold" if ctrl_metrics.get('h3_predictive_gain', 0) > 0 else "text-danger"),
                ]),
                html.Tr([
                    html.Td("Bivariate Moran's I (spatial predictive alignment)"),
                    html.Td(f"{hyp_metrics.get('h3_moran_i', 0.0):.4f}"),
                    html.Td(f"{ctrl_metrics.get('h3_moran_i', 0.0):.4f}"),
                ]),
                html.Tr([
                    html.Td("Moran's I Significance (p-value)"),
                    html.Td(f"{hyp_metrics.get('h3_moran_pvalue', 1.0):.4f}", className="text-success fw-bold" if hyp_metrics.get('h3_moran_pvalue', 1) < 0.05 else ""),
                    html.Td(f"{ctrl_metrics.get('h3_moran_pvalue', 1.0):.4f}"),
                ]),
            ]
            h3_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Predictive Validation Metric"),
                    html.Th("Hypothesis Run"),
                    html.Th("Control Run"),
                ])),
                html.Tbody(h3_rows)
            ], className="table table-striped table-sm small border")

            report_children.append(html.Div([
                html.H5("H3: Predictive Capacity Details", className="mt-4 text-primary"),
                html.P("Hypothesis 3 tests forecasting capacity using a 70/30 time split. The model must beat the baseline (gain > 0) and exhibit statistically significant spatial alignment (Moran's I p-value < 0.05).", className="small text-muted"),
                h3_table
            ]))

            return html.Div(report_children)

        except Exception as e:
            logger.error(f"Error generating hypothesis report: {e}", exc_info=True)
            return dbc.Alert(f"An error occurred while compiling the report: {e}", color="danger", className="mt-3")
