"""Convergence panel callbacks for the EDEL dashboard."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from edel.io.artifact import make_stage_artifact
from edel.experiments.registry import list_experiments, get_experiment
from edel.analysis.convergence import run_convergence_analysis

logger = logging.getLogger(__name__)


def _get_configs_with_embeddings(base_path: Path) -> list[str]:
    """Scan all experiments and return those that have completed embeddings/clustering."""
    available = []
    for name in list_experiments():
        try:
            cfg = get_experiment(name)
            art_emb = make_stage_artifact(cfg, base_path, "embeddings", "embeddings")
            art_clust = make_stage_artifact(cfg, base_path, "clustering", "clustering")
            if art_emb.parquet_path.exists() or art_clust.parquet_path.exists():
                available.append(name)
        except Exception:
            continue
    return sorted(available)


def register_convergence_callbacks(app: Dash, base_path: Path) -> None:
    """Register callbacks for the convergence analysis panel."""
    
    # Callback to populate the experiment run selection dropdown
    @app.callback(
        Output("conv-experiment-select", "options"),
        [Input("artifact-update-store", "data")]
    )
    def update_experiment_dropdown(update_val):
        try:
            available = _get_configs_with_embeddings(base_path)
            return [{"label": name, "value": name} for name in available]
        except Exception as e:
            logger.error(f"Error populating convergence selector dropdown: {e}")
            return []

    # Callback to execute the convergence analysis and build the report
    @app.callback(
        Output("conv-report-container", "children"),
        [Input("btn-run-conv-analysis", "n_clicks")],
        [State("conv-experiment-select", "value")],
        prevent_initial_call=True
    )
    def generate_convergence_report(n_clicks, experiment_id):
        if not n_clicks:
            return html.Div(
                "Select a run from the dropdown and click 'Run Convergence Analysis'. Cached results will load instantly.",
                className="text-center text-muted p-5 my-4 border rounded bg-light"
            )
            
        if not experiment_id:
            return dbc.Alert("Please select an experiment run.", color="warning", className="mt-3")

        try:
            # Execute convergence analysis (uses caching internally)
            results = run_convergence_analysis(experiment_id, base_path)
            
            N = results["N"]
            h1_res = results["h1_results"]
            h2_res = results["h2_results"]
            h3_res = results["h3_results"]
            
            h1_sizes = h1_res["sample_sizes"]
            h2_sizes = h2_res["sample_sizes"]
            h3_percentages = h3_res["percentages"]
            h3_sizes = h3_res["sizes"]
            
            # -------------------------------------------------------------------
            # 1. Compute recommended sizes
            # -------------------------------------------------------------------
            rec_h1 = "10,000 papers"
            for sz in h1_sizes:
                # Average Wasserstein distance and average KS stats delta across all transitions
                w_vals = []
                ks_deltas = []
                for key in h1_res["full_refs"].keys():
                    w_vals.append(np.mean(h1_res["data"][sz]["w_dist"][key]))
                    ks_deltas.append(abs(np.mean(h1_res["data"][sz]["ks_stat"][key]) - h1_res["full_refs"][key]))
                avg_w = np.mean(w_vals)
                avg_ks_delta = np.mean(ks_deltas)
                if avg_w < 0.08 and avg_ks_delta < 0.05:
                    rec_h1 = f"{sz:,} papers"
                    break
            if rec_h1 == "10,000 papers" and N < 10000:
                rec_h1 = f"Full corpus ({N:,} papers)"
                    
            rec_h2 = "10,000 papers"
            for sz in h2_sizes:
                avg_mae_z = np.mean(h2_res["data"][sz]["mae_z"])
                avg_jaccard = np.mean(h2_res["data"][sz]["jaccard"])
                # The z-score estimate has an inherent standard deviation of ~1.0 due to 
                # query selection sampling noise (max_queries = 25). Therefore, we combine 
                # MAE (< 0.8) with Jaccard operator overlap stabilization (>= 0.80).
                if avg_mae_z < 0.8 or avg_jaccard >= 0.80:
                    rec_h2 = f"{sz:,} papers"
                    break
            if rec_h2 == "10,000 papers" and N < 10000:
                rec_h2 = f"Full corpus ({N:,} papers)"
                    
            rec_h3 = "50% of corpus"
            full_gain = h3_res["full_gain"]
            for pct, sz in zip(h3_percentages, h3_sizes):
                # Calculate if Scheme B (Temporal Stratified) is within 10% of full gain
                b_gain = np.mean(h3_res["data"]["Scheme B"][sz])
                if abs(b_gain - full_gain) < 0.02 or (full_gain > 0 and abs(b_gain - full_gain)/full_gain < 0.1):
                    rec_h3 = f"{int(pct * 100)}% of corpus ({sz:,} papers)"
                    break
                    
            # -------------------------------------------------------------------
            # 2. Build Plots
            # -------------------------------------------------------------------
            
            # --- H1 Plot 1: 1D Wasserstein distance to full observed distribution ---
            fig_h1_w = go.Figure()
            # Draw average across all transitions for each sample size
            h1_x = h1_sizes
            h1_w_means = []
            h1_w_stds = []
            for sz in h1_sizes:
                all_rep_w = []
                for rep in range(20):
                    # Average over 6 transitions
                    rep_val = np.mean([h1_res["data"][sz]["w_dist"][k][rep] for k in h1_res["full_refs"].keys()])
                    all_rep_w.append(rep_val)
                h1_w_means.append(np.mean(all_rep_w))
                h1_w_stds.append(np.std(all_rep_w))
                
            h1_w_means = np.array(h1_w_means)
            h1_w_stds = np.array(h1_w_stds)
            
            fig_h1_w.add_trace(go.Scatter(
                x=h1_x + h1_x[::-1],
                y=(h1_w_means + h1_w_stds).tolist() + (h1_w_means - h1_w_stds).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 123, 255, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name="±1 SD"
            ))
            fig_h1_w.add_trace(go.Scatter(
                x=h1_x, y=h1_w_means,
                mode="lines+markers",
                line=dict(color="#007bff", width=3),
                name="Average 1D Wasserstein"
            ))
            fig_h1_w.update_layout(
                title="H1 Distribution Convergence: 1D Wasserstein Distance to Full Dataset",
                xaxis_title="Sample Size (n)",
                yaxis_title="Wasserstein Distance (Lower = Closer to Full)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # --- H1 Plot 2: KS Statistic convergence error ---
            fig_h1_ks = go.Figure()
            h1_ks_errors = []
            h1_ks_stds = []
            for sz in h1_sizes:
                all_rep_errs = []
                for rep in range(20):
                    # Mean absolute difference in KS statistic across 6 transitions
                    rep_err = np.mean([
                        abs(h1_res["data"][sz]["ks_stat"][k][rep] - h1_res["full_refs"][k])
                        for k in h1_res["full_refs"].keys()
                    ])
                    all_rep_errs.append(rep_err)
                h1_ks_errors.append(np.mean(all_rep_errs))
                h1_ks_stds.append(np.std(all_rep_errs))
                
            h1_ks_errors = np.array(h1_ks_errors)
            h1_ks_stds = np.array(h1_ks_stds)
            
            fig_h1_ks.add_trace(go.Scatter(
                x=h1_x + h1_x[::-1],
                y=(h1_ks_errors + h1_ks_stds).tolist() + (h1_ks_errors - h1_ks_stds).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(40, 167, 69, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name="±1 SD"
            ))
            fig_h1_ks.add_trace(go.Scatter(
                x=h1_x, y=h1_ks_errors,
                mode="lines+markers",
                line=dict(color="#28a745", width=3),
                name="KS Stat Absolute Error"
            ))
            fig_h1_ks.update_layout(
                title="H1 Statistic Convergence: Mean Absolute Error of KS Statistic",
                xaxis_title="Sample Size (n)",
                yaxis_title="MAE of KS Statistic",
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40)
            )

            # --- H2 Plot 1: MAE of transition z-scores ---
            fig_h2_mae = go.Figure()
            h2_x = h2_sizes
            h2_mae_means = [np.mean(h2_res["data"][sz]["mae_z"]) for sz in h2_sizes]
            h2_mae_stds = [np.std(h2_res["data"][sz]["mae_z"]) for sz in h2_sizes]
            
            h2_mae_means = np.array(h2_mae_means)
            h2_mae_stds = np.array(h2_mae_stds)
            
            fig_h2_mae.add_trace(go.Scatter(
                x=h2_x + h2_x[::-1],
                y=(h2_mae_means + h2_mae_stds).tolist() + (h2_mae_means - h2_mae_stds).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(220, 53, 69, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name="±1 SD"
            ))
            fig_h2_mae.add_trace(go.Scatter(
                x=h2_x, y=h2_mae_means,
                mode="lines+markers",
                line=dict(color="#dc3545", width=3),
                name="Mean Absolute Error (MAE) of z-scores"
            ))
            fig_h2_mae.update_layout(
                title="H2 z-score Convergence: MAE compared to Full Dataset z-scores",
                xaxis_title="Sample Size (n)",
                yaxis_title="z-score MAE (Lower = Closer to Full)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40)
            )

            # --- H2 Plot 2: Jaccard overlap of significant operators ---
            fig_h2_jac = go.Figure()
            h2_jac_means = [np.mean(h2_res["data"][sz]["jaccard"]) for sz in h2_sizes]
            h2_jac_stds = [np.std(h2_res["data"][sz]["jaccard"]) for sz in h2_sizes]
            
            h2_jac_means = np.array(h2_jac_means)
            h2_jac_stds = np.array(h2_jac_stds)
            
            fig_h2_jac.add_trace(go.Scatter(
                x=h2_x + h2_x[::-1],
                y=(h2_jac_means + h2_jac_stds).tolist() + (h2_jac_means - h2_jac_stds).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(23, 162, 184, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name="±1 SD"
            ))
            fig_h2_jac.add_trace(go.Scatter(
                x=h2_x, y=h2_jac_means,
                mode="lines+markers",
                line=dict(color="#17a2b8", width=3),
                name="Jaccard Overlap"
            ))
            fig_h2_jac.update_layout(
                title="H2 Operator Overlap: Jaccard Similarity of Significant Operators (p < 0.05)",
                xaxis_title="Sample Size (n)",
                yaxis_title="Jaccard Index (Higher = More Consistent)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40)
            )

            # --- H3 Plot 1: Predictive Gain vs Sample size/fraction ---
            fig_h3_gain = go.Figure()
            # X-axis will be sample sizes
            h3_x = h3_sizes
            h3_x_labels = [f"{sz:,} ({pct*100:.0f}%)" for sz, pct in zip(h3_sizes, h3_percentages)]
            
            # Scheme A (Uniform)
            a_means = [np.mean(h3_res["data"]["Scheme A"][sz]) for sz in h3_sizes]
            a_stds = [np.std(h3_res["data"]["Scheme A"][sz]) for sz in h3_sizes]
            a_means = np.array(a_means)
            a_stds = np.array(a_stds)
            
            # Scheme B (Temporal Stratified)
            b_means = [np.mean(h3_res["data"]["Scheme B"][sz]) for sz in h3_sizes]
            b_stds = [np.std(h3_res["data"]["Scheme B"][sz]) for sz in h3_sizes]
            b_means = np.array(b_means)
            b_stds = np.array(b_stds)
            
            # Plot Scheme A
            fig_h3_gain.add_trace(go.Scatter(
                x=h3_x + h3_x[::-1],
                y=(a_means + a_stds).tolist() + (a_means - a_stds).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 193, 7, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name="Scheme A ±1 SD"
            ))
            fig_h3_gain.add_trace(go.Scatter(
                x=h3_x, y=a_means,
                mode="lines+markers",
                line=dict(color="#ffc107", width=3),
                name="Scheme A: Uniform Random"
            ))
            
            # Plot Scheme B
            fig_h3_gain.add_trace(go.Scatter(
                x=h3_x + h3_x[::-1],
                y=(b_means + b_stds).tolist() + (b_means - b_stds).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(111, 66, 193, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name="Scheme B ±1 SD"
            ))
            fig_h3_gain.add_trace(go.Scatter(
                x=h3_x, y=b_means,
                mode="lines+markers",
                line=dict(color="#6f42c1", width=3),
                name="Scheme B: Temporal Stratified"
            ))
            
            # Plot Ground Truth (Full dataset gain)
            fig_h3_gain.add_trace(go.Scatter(
                x=[h3_x[0], h3_x[-1]], y=[full_gain, full_gain],
                mode="lines",
                line=dict(color="#343a40", width=2, dash="dash"),
                name=f"Full Corpus Gain ({full_gain:.4f})"
            ))
            
            fig_h3_gain.update_layout(
                title="H3 Forecasting Calibration: Predictive Gain by Sampling Scheme",
                xaxis=dict(
                    tickmode='array',
                    tickvals=h3_x,
                    ticktext=h3_x_labels,
                    title="Sample Size (Fraction %)"
                ),
                yaxis_title="Predictive Gain (W_base - W_edel)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # --- H3 Plot 2: Absolute error of gain |Gain_n - Gain_full| ---
            fig_h3_err = go.Figure()
            a_errs = [abs(np.mean(h3_res["data"]["Scheme A"][sz]) - full_gain) for sz in h3_sizes]
            b_errs = [abs(np.mean(h3_res["data"]["Scheme B"][sz]) - full_gain) for sz in h3_sizes]
            
            fig_h3_err.add_trace(go.Scatter(
                x=h3_x, y=a_errs,
                mode="lines+markers",
                line=dict(color="#ffc107", width=3),
                name="Scheme A: Uniform Error"
            ))
            fig_h3_err.add_trace(go.Scatter(
                x=h3_x, y=b_errs,
                mode="lines+markers",
                line=dict(color="#6f42c1", width=3),
                name="Scheme B: Temporal Stratified Error"
            ))
            
            fig_h3_err.update_layout(
                title="H3 Calibration Convergence: Absolute Gain Error to Full Corpus",
                xaxis=dict(
                    tickmode='array',
                    tickvals=h3_x,
                    ticktext=h3_x_labels,
                    title="Sample Size (Fraction %)"
                ),
                yaxis_title="Absolute Gain Error |Gain_n - Gain_full| (Lower = Better)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # -------------------------------------------------------------------
            # 3. Assemble Layout Report
            # -------------------------------------------------------------------
            
            summary_cards = dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("H1 Target Size", className="card-title text-primary fw-bold"),
                            html.H3(rec_h1, className="card-text fw-bold text-dark"),
                            html.P("Recommended for structural transition validation.", className="small text-muted mb-0")
                        ])
                    ], className="border-start border-primary border-4 shadow-sm mb-3"), md=4
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("H2 Target Size", className="card-title text-danger fw-bold"),
                            html.H3(rec_h2, className="card-text fw-bold text-dark"),
                            html.P("Recommended for local transition z-score stabilization.", className="small text-muted mb-0")
                        ])
                    ], className="border-start border-danger border-4 shadow-sm mb-3"), md=4
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("H3 Calibration Target", className="card-title text-success fw-bold"),
                            html.H3(rec_h3, className="card-text fw-bold text-dark"),
                            html.P("Optimal sample fraction using Temporal Stratified sampling.", className="small text-muted mb-0")
                        ])
                    ], className="border-start border-success border-4 shadow-sm mb-3"), md=4
                ),
            ], className="g-3 mb-4")

            return html.Div([
                # Title segment
                html.Div([
                    html.H4(f"Convergence Calibration Report: {experiment_id}", className="fw-bold text-dark mb-1"),
                    html.P(f"Reference corpus size N = {N:,} papers.", className="text-muted small mb-3")
                ]),
                
                # Recommendations Summary
                summary_cards,
                
                # Plot Tab Grid
                dbc.Tabs([
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=fig_h1_w), md=6),
                            dbc.Col(dcc.Graph(figure=fig_h1_ks), md=6)
                        ], className="mt-3")
                    ], label="H1 Convergence", tab_id="tab-conv-h1"),
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=fig_h2_mae), md=6),
                            dbc.Col(dcc.Graph(figure=fig_h2_jac), md=6)
                        ], className="mt-3")
                    ], label="H2 Convergence", tab_id="tab-conv-h2"),
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=fig_h3_gain), md=6),
                            dbc.Col(dcc.Graph(figure=fig_h3_err), md=6)
                        ], className="mt-3")
                    ], label="H3 Calibration", tab_id="tab-conv-h3")
                ], id="conv-results-tabs", active_tab="tab-conv-h1")
            ])
            
        except Exception as e:
            logger.exception("Error during convergence analysis execution")
            return dbc.Alert(f"An error occurred during analysis: {str(e)}", color="danger", className="mt-3")
