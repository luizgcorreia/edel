"""Report Generator callbacks for the EDEL Dashboard."""

import io
import logging
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from edel.experiments.runner import load_registry
from edel.experiments.analyzer import analyze_experiments
from edel.dashboard.cache import get_results_df

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions (mirrors edel.experiments.report)
# ---------------------------------------------------------------------------

META_COLUMNS = [
    "experiment_id", "provider", "topic_name",
    "n_documents_requested", "embedding_model", "projection_method",
]

_TRANSITIONS = ["pm", "pf", "pi", "mp", "mf", "mi", "fp", "fm", "fi", "ip", "im", "if"]
_PAIRS = ["pm", "mf", "fi", "pf", "pi", "mi"]

HYPOTHESIS_COLUMNS: dict[str, dict[str, list[str]]] = {
    "H1": {
        "Primary": ["h1_energy_stat", "h1_energy_pvalue"],
        "Wasserstein Effect Sizes": [
            "h1_w_norm_pm", "h1_w_norm_mf", "h1_w_norm_fi",
            "h1_w_cos_pm_mf", "h1_w_cos_pm_fi", "h1_w_cos_mf_fi",
        ],
        "KS (stat)": [
            "h1_ks_stat_norm_pm", "h1_ks_stat_norm_mf", "h1_ks_stat_norm_fi",
            "h1_ks_stat_cos_pm_mf", "h1_ks_stat_cos_pm_fi", "h1_ks_stat_cos_mf_fi",
        ],
        "KS (pvalue)": [
            "h1_ks_pvalue_norm_pm", "h1_ks_pvalue_norm_mf", "h1_ks_pvalue_norm_fi",
            "h1_ks_pvalue_cos_pm_mf", "h1_ks_pvalue_cos_pm_fi", "h1_ks_pvalue_cos_mf_fi",
        ],
    },
    "H2": {
        "Primary (p-values)": [f"h2_pvalue_{t}" for t in _TRANSITIONS],
        "Wasserstein Distance": [f"h2_w_dist_{t}" for t in _TRANSITIONS],
        "Z-scores": [f"h2_z_{t}" for t in _TRANSITIONS],
        "Asymmetry (diff)": [f"h2b_diff_{p}" for p in _PAIRS],
        "Asymmetry (p-value)": [f"h2b_pvalue_{p}" for p in _PAIRS],
    },
    "H3": {
        "Primary": [
            "h3_w_edel", "h3_w_baseline", "h3_predictive_gain",
            "h3_gain_pvalue", "h3_moran_i", "h3_moran_pvalue",
        ],
    },
}


def _all_hypothesis_columns(hypothesis: str) -> list[str]:
    groups = HYPOTHESIS_COLUMNS.get(hypothesis, {})
    return [c for cols in groups.values() for c in cols]


def _build_excel_bytes(df: pd.DataFrame, hypotheses: list[str]) -> bytes:
    """Build an Excel workbook in-memory and return bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for ht in hypotheses:
            if ht not in HYPOTHESIS_COLUMNS:
                continue

            all_cols = _all_hypothesis_columns(ht)
            available = [c for c in all_cols if c in df.columns]
            meta_available = [c for c in META_COLUMNS if c in df.columns]

            if not available:
                continue

            tab_cols = meta_available + [c for c in available if c not in set(meta_available)]
            tab_df = df[tab_cols].copy()

            if ht == "H1" and "h1_energy_pvalue" in tab_df.columns:
                tab_df["h1_pass"] = tab_df["h1_energy_pvalue"] < 0.05
            elif ht == "H2":
                pval_cols = [c for c in tab_df.columns if c.startswith("h2_pvalue_") and c != "h2_pvalue"]
                if pval_cols:
                    tab_df["h2_num_passed"] = (tab_df[pval_cols] < 0.05).sum(axis=1)
                    tab_df["h2_pass"] = tab_df["h2_num_passed"] >= 3
            elif ht == "H3" and "h3_predictive_gain" in tab_df.columns and "h3_gain_pvalue" in tab_df.columns:
                tab_df["h3_pass"] = (tab_df["h3_predictive_gain"] > 0) & (tab_df["h3_gain_pvalue"] < 0.05)

            tab_df.to_excel(writer, sheet_name=ht, index=False)

    output.seek(0)
    return output.getvalue()


def register_report_generator_callbacks(app: Dash, base_path: Path) -> None:

    # ── Populate experiment dropdown from registry ────────────────────────
    @app.callback(
        Output("report-experiment-select", "options"),
        Input("artifact-update-store", "data"),
    )
    def update_experiment_dropdown(update_val):
        try:
            registry = load_registry(base_path)
            ids = sorted({exp["experiment_id"] for exp in registry})
            return [{"label": eid, "value": eid} for eid in ids]
        except Exception as e:
            logger.error(f"Error updating report experiment dropdown: {e}")
            return []

    # ── Select All / None toggle ──────────────────────────────────────────
    @app.callback(
        Output("report-experiment-select", "value"),
        Input("report-toggle-all-btn", "n_clicks"),
        State("report-experiment-select", "options"),
        State("report-experiment-select", "value"),
        prevent_initial_call=True,
    )
    def toggle_select_all(n_clicks, options, current_value):
        if not options:
            return no_update
        all_values = {o["value"] for o in options}
        if current_value and set(current_value) == all_values:
            return []
        return sorted(all_values)

    # ── Generate report + download ────────────────────────────────────────
    @app.callback(
        Output("report-download-component", "data"),
        Output("report-status-msg", "children"),
        Output("report-preview-container", "children"),
        Input("btn-generate-report", "n_clicks"),
        State("report-experiment-select", "value"),
        State("report-hypothesis-checklist", "value"),
        State("report-mode-radio", "value"),
        prevent_initial_call=True,
    )
    def handle_generate_report(n_clicks, exp_ids, hypotheses, mode):
        if not n_clicks:
            return no_update, "", html.Div()

        if not exp_ids:
            return no_update, (
                html.Span("⚠️ Please select at least one experiment.", className="text-danger")
            ), no_update

        if not hypotheses:
            return no_update, (
                html.Span("⚠️ Please select at least one hypothesis.", className="text-danger")
            ), no_update

        try:
            # ── Get or compute results DataFrame ──────────────────────────
            if mode == "force":
                registry = load_registry(base_path)
                id_set = set(exp_ids)
                records = [r for r in registry if r["experiment_id"] in id_set]
                missing = id_set - {r["experiment_id"] for r in records}
                if missing:
                    logger.warning(f"Experiments not in registry: {missing}")
                if not records:
                    return no_update, (
                        html.Span("⚠️ No matching experiments found in registry.", className="text-danger")
                    ), no_update
                df = analyze_experiments(records, base_path=base_path)
            else:
                df = get_results_df(base_path)
                if df.empty:
                    return no_update, (
                        html.Span("⚠️ Cache is empty. Rebuild on the Metrics Analysis tab or use 'Recompute' mode.", className="text-danger")
                    ), no_update
                df = df[df["experiment_id"].isin(exp_ids)]

            if df.empty:
                return no_update, (
                    html.Span("⚠️ No results for selected experiments.", className="text-danger")
                ), no_update

            # ── Build Excel bytes ──────────────────────────────────────────
            excel_bytes = _build_excel_bytes(df, hypotheses)

            # ── Preview summary ───────────────────────────────────────────
            preview = html.Div([
                html.H6("Report generated:", className="mb-2"),
                html.Ul([
                    html.Li(f"Experiments: {', '.join(sorted(df['experiment_id'].unique()))}"),
                    html.Li(f"Hypotheses: {', '.join(hypotheses)}"),
                    html.Li(f"Mode: {'Recomputed from artifacts' if mode == 'force' else 'Cached results'}"),
                    html.Li(f"Rows: {len(df)}, Columns: {len(df.columns)}"),
                ], className="mb-0"),
            ])

            status = html.Span("✅ Report ready. Download should start automatically.", className="text-success")

            return (
                dcc.send_bytes(excel_bytes, f"edel_report_{'_'.join(hypotheses)}.xlsx"),
                status,
                preview,
            )

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            return no_update, (
                html.Span(f"❌ Error: {e}", className="text-danger")
            ), no_update
