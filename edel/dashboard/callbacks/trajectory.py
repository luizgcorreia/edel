"""Callbacks for the Trajectory Explorer panel (Tab 6)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Dash, Input, Output, State, callback_context, html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from edel.experiments.registry import get_experiment
from edel.io.artifact import make_stage_artifact, load_artifact
from edel.analysis.trajectory import analyze_trajectory, format_report, ASPECTS

logger = logging.getLogger(__name__)

# Simple in-process cache so we don't re-parse the huge parquet on every click
_DF_CACHE: dict[str, pd.DataFrame] = {}


def _load_dr_df(experiment_name: str, base_path: Path) -> pd.DataFrame:
    """Load (and cache) the dimensionality-reduction parquet for an experiment."""
    if experiment_name in _DF_CACHE:
        return _DF_CACHE[experiment_name]

    config = get_experiment(experiment_name)
    # The DR artifact is saved under the "clustering" stage (includes projections)
    try:
        art = make_stage_artifact(config, base_path, "clustering", "clustering")
        df = load_artifact(art)
    except Exception:
        # Fall back to DR stage artifact
        art = make_stage_artifact(config, base_path, "dimensionality_reduction", "dr")
        df = load_artifact(art)

    _DF_CACHE[experiment_name] = df
    return df


# ---------------------------------------------------------------------------
# Result card builder
# ---------------------------------------------------------------------------

def _segment_badge(label: str, text: str, color: str = "secondary") -> html.Div:
    """Collapsible card for a single segment."""
    slug = label.lower().replace(" ", "-")
    return html.Div([
        dbc.Button(
            [dbc.Badge(label, color=color, className="me-2"), "▼"],
            id={"type": "seg-toggle", "index": slug},
            color="link",
            size="sm",
            className="text-start p-0 mb-1",
            style={"textDecoration": "none", "color": "inherit"},
            n_clicks=0,
        ),
        dbc.Collapse(
            html.P(text or "(empty)", className="small text-muted border-start ps-2 mt-1"),
            id={"type": "seg-collapse", "index": slug},
            is_open=False,
        )
    ])


def _neighbor_card(n: dict, rank: int, aspect: str) -> dbc.Card:
    """Build a rich neighbour card with expandable segments."""
    year = n.get("publication_year", "")
    cits = n.get("cited_by_count", "")
    doi = n.get("doi", "")
    openalex_url = n.get("id", "")
    dist = n["distance"]

    header_children = [
        html.Span(f"{rank}. ", className="text-muted me-1"),
        html.Strong(n.get("title", "Unknown")),
        dbc.Badge(f"{dist:.4f}", color="info", className="ms-2 float-end"),
    ]
    if year:
        header_children.insert(2, html.Small(f" ({year})", className="text-muted"))

    # Segment rows — show current aspect segment first and prominently
    segment_items = [
        dbc.ListGroupItem([
            html.Strong(f"{asp.capitalize()}: "),
            html.Span(n.get(asp, "(empty)"), className="small text-muted")
        ], color="info" if asp == aspect else None)
        for asp in ASPECTS
    ]

    # Special "Opened Problem" highlight for interpretation neighbours
    opened_problem_section = []
    if aspect == "interpretation":
        prob_text = n.get("problem", "")
        if prob_text and str(prob_text).strip():
            opened_problem_section = [
                dbc.Alert([
                    html.Strong("⬆ Opened Problem: "),
                    html.Span(prob_text, className="small")
                ], color="warning", className="mt-2 mb-0 py-2")
            ]

    links = []
    if openalex_url:
        links.append(html.A("OpenAlex", href=openalex_url, target="_blank",
                            className="btn btn-sm btn-outline-secondary me-1"))
    if doi:
        links.append(html.A("DOI", href=f"https://doi.org/{doi}", target="_blank",
                            className="btn btn-sm btn-outline-secondary"))

    return dbc.Card([
        dbc.CardHeader(header_children, className="py-2"),
        dbc.CardBody([
            # Abstract
            dbc.Accordion([
                dbc.AccordionItem(
                    html.P(n.get("abstract_text", "(no abstract)"), className="small"),
                    title="Abstract",
                )
            ], start_collapsed=True, flush=True, className="mb-2"),
            # Segments
            dbc.Accordion([
                dbc.AccordionItem(
                    dbc.ListGroup(segment_items, flush=True),
                    title="Epistemic Segments",
                )
            ], start_collapsed=True, flush=True, className="mb-2"),
            *opened_problem_section,
            html.Div(links, className="mt-2"),
        ], className="py-2")
    ], className="mb-2", style={"fontSize": "0.85rem"})


def _aspect_accordion_item(asp: str, asp_result: dict, target_segment: str) -> dbc.AccordionItem:
    """Build an accordion item for one aspect's results."""
    neighbors = asp_result.get("neighbors", [])
    error = asp_result.get("error")

    n_count = len(neighbors)
    badge_color = "success" if n_count > 0 else "secondary"
    title = html.Span([
        asp.capitalize(),
        dbc.Badge(f"{n_count} neighbours", color=badge_color, className="ms-2"),
    ])

    body = [
        dbc.Alert([
            html.Strong("Target Segment: "),
            html.Span(target_segment or "(empty)", className="small")
        ], color="dark", className="py-2 mb-3"),
    ]

    if error:
        body.append(dbc.Alert(f"⚠ {error}", color="warning"))
    elif not neighbors:
        body.append(html.P("No neighbours found.", className="text-muted"))
    else:
        body.extend([_neighbor_card(n, i + 1, asp) for i, n in enumerate(neighbors)])

    return dbc.AccordionItem(html.Div(body), title=title)


# ---------------------------------------------------------------------------
# Trajectory plot builder
# ---------------------------------------------------------------------------

def _build_trajectory_plot(df: pd.DataFrame, result: dict, method: str) -> go.Figure:
    """Build the 2D scatter plot with trajectory arrows.

    For Work ID mode: overlay the four projection points connected by arrows.
    For synthetic mode: just show the scatter background (no trajectory).
    """
    target = result["target"]
    aspects_data = result["aspects"]

    # Background scatter (sample for speed)
    sample_size = min(2000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    x_col = f"proj_problem_{method}_x"
    y_col = f"proj_problem_{method}_y"

    fig = go.Figure()

    if x_col in df_sample.columns and y_col in df_sample.columns:
        fig.add_trace(go.Scatter(
            x=df_sample[x_col],
            y=df_sample[y_col],
            mode="markers",
            marker=dict(size=3, color="rgba(100,120,200,0.25)"),
            name="All Papers",
            hoverinfo="skip",
        ))

    # Trajectory path (only for Work ID where coords exist)
    if result["mode"] == "paper_id":
        traj_xs, traj_ys, traj_labels = [], [], []
        for asp in ASPECTS:
            coords = aspects_data.get(asp, {}).get("target_vec_2d")
            if coords:
                traj_xs.append(coords[0])
                traj_ys.append(coords[1])
                traj_labels.append(asp.capitalize())

        if traj_xs:
            # Lines
            fig.add_trace(go.Scatter(
                x=traj_xs, y=traj_ys,
                mode="lines+markers+text",
                line=dict(color="gold", width=2),
                marker=dict(size=10, color="gold", symbol="circle"),
                text=traj_labels,
                textposition="top center",
                textfont=dict(family="Arial Black", size=11, color="white"),
                name="Trajectory",
            ))

            # Arrows between consecutive positions
            for i in range(len(traj_xs) - 1):
                fig.add_annotation(
                    x=traj_xs[i + 1], y=traj_ys[i + 1],
                    ax=traj_xs[i], ay=traj_ys[i],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3, arrowsize=1.5,
                    arrowwidth=2, arrowcolor="gold",
                )

    # Neighbours overlay (first aspect with 2D coords)
    for asp in ASPECTS:
        asp_result = aspects_data.get(asp, {})
        neighbours = asp_result.get("neighbors", [])
        asp_x_col = f"proj_{asp}_{method}_x"
        asp_y_col = f"proj_{asp}_{method}_y"
        n_xs, n_ys, n_texts = [], [], []
        for n in neighbours:
            n_data = df[df["id"] == n["id"]]
            if n_data.empty:
                continue
            n_row = n_data.iloc[0]
            if asp_x_col in df.columns and pd.notna(n_row.get(asp_x_col)):
                n_xs.append(float(n_row[asp_x_col]))
                n_ys.append(float(n_row[asp_y_col]))
                n_texts.append(n.get("title", "")[:40])
        if n_xs:
            fig.add_trace(go.Scatter(
                x=n_xs, y=n_ys,
                mode="markers",
                marker=dict(size=8, color="tomato", symbol="diamond", opacity=0.7),
                text=n_texts,
                hoverinfo="text",
                name=f"{asp.capitalize()} neighbours",
            ))
            break  # Only first aspect to avoid clutter; full detail is in the cards

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_trajectory_callbacks(app: Dash, base_path: Path) -> None:

    # --- Toggle input mode ---
    @app.callback(
        [Output("traj-workid-input-group", "style"),
         Output("traj-synthetic-input-group", "style"),
         Output("traj-mode-id-btn", "outline"),
         Output("traj-mode-synthetic-btn", "outline"),
         Output("traj-input-mode", "data")],
        [Input("traj-mode-id-btn", "n_clicks"),
         Input("traj-mode-synthetic-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def toggle_input_mode(id_clicks, synthetic_clicks):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if btn_id == "traj-mode-id-btn":
            return (
                {"display": "block"}, {"display": "none"},
                False, True, "work_id"
            )
        else:
            return (
                {"display": "none"}, {"display": "block"},
                True, False, "synthetic"
            )

    # --- Enable Run button when experiment is selected ---
    @app.callback(
        Output("traj-run-btn", "disabled"),
        Input("traj-experiment-select", "value"),
    )
    def toggle_run_button(experiment_name):
        return not bool(experiment_name)

    # --- Main analysis callback ---
    @app.callback(
        [Output("traj-plot", "figure"),
         Output("traj-results-container", "children"),
         Output("traj-status-msg", "children"),
         Output("traj-download-btn", "style"),
         Output("traj-results-store", "data"),
         Output("traj-metrics-container", "children")],
        Input("traj-run-btn", "n_clicks"),
        [State("traj-experiment-select", "value"),
         State("traj-method-select", "value"),
         State("traj-space-select", "value"),
         State("traj-input-mode", "data"),
         State("traj-paper-id", "value"),
         State("traj-segment-problem", "value"),
         State("traj-segment-method", "value"),
         State("traj-segment-finding", "value"),
         State("traj-segment-interpretation", "value"),
         State("traj-k", "value"),
         State("traj-radius", "value")],
        prevent_initial_call=True,
    )
    def run_trajectory_analysis(
        n_clicks, experiment_name, method, space, input_mode,
        paper_id, seg_problem, seg_method, seg_finding, seg_interpretation,
        k, radius
    ):
        if not n_clicks or not experiment_name:
            raise PreventUpdate

        k = int(k) if k else 5
        radius = float(radius) if radius else None

        try:
            df = _load_dr_df(experiment_name, base_path)
        except Exception as e:
            return dash.no_update, dash.no_update, f"Failed to load dataset: {e}", {"display": "none"}, None, dash.no_update

        try:
            if input_mode == "work_id":
                if not paper_id or not paper_id.strip():
                    return dash.no_update, dash.no_update, "Please enter a Work ID.", {"display": "none"}, None, dash.no_update
                paper_id = paper_id.strip()
                if paper_id not in df["id"].values:
                    return dash.no_update, dash.no_update, f"Work ID not found: {paper_id}", {"display": "none"}, None, dash.no_update

                result = analyze_trajectory(
                    df, paper_id=paper_id,
                    space=space, method=method, k=k, radius=radius,
                )

            else:
                # Synthetic mode — call the embedding API
                segments = {
                    "problem": (seg_problem or "").strip(),
                    "method": (seg_method or "").strip(),
                    "finding": (seg_finding or "").strip(),
                    "interpretation": (seg_interpretation or "").strip(),
                }
                if not any(segments.values()):
                    return dash.no_update, dash.no_update, "Please enter at least one text segment.", {"display": "none"}, None, dash.no_update

                # Load config and embedding client
                config = get_experiment(experiment_name)
                from edel.io.llm import get_llm_client
                embed_cfg = config.get("embedding", {})
                client = get_llm_client(embed_cfg)

                embedding_vectors: dict[str, np.ndarray] = {}
                for asp, text in segments.items():
                    if text:
                        raw = client.generate_embedding(text)
                        embedding_vectors[asp] = np.array(raw, dtype=np.float32)

                result = analyze_trajectory(
                    df,
                    segments=segments,
                    embedding_vectors=embedding_vectors,
                    space="embedding",   # always embedding for synthetic
                    method=method,
                    k=k,
                    radius=radius,
                )

        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            return dash.no_update, dash.no_update, f"Analysis error: {e}", {"display": "none"}, None, dash.no_update

        # Build plot
        try:
            fig = _build_trajectory_plot(df, result, method)
        except Exception as e:
            fig = go.Figure()
            logger.warning("Could not build trajectory plot: %s", e)

        # Build result accordion
        target = result["target"]
        accordion_items = [
            _aspect_accordion_item(asp, result["aspects"].get(asp, {}), target.get(asp, ""))
            for asp in ASPECTS
        ]

        target_header = dbc.Alert([
            html.H6([
                html.Strong(target.get("title", "(Synthetic Paper)")),
                dbc.Badge(
                    f"{target.get('publication_year', '')} · {target.get('cited_by_count', '')} citations",
                    color="dark", className="ms-2"
                ),
            ], className="mb-1"),
        ], color="dark", className="mb-3")

        results_children = [
            target_header,
            dbc.Accordion(accordion_items, start_collapsed=False, always_open=True),
        ]

        # Prepare serializable result for store (convert numpy arrays/scalars to lists/native types)
        def _json_serial(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, (pd.Timestamp, pd.Series, pd.DataFrame)):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        store_result = json.loads(json.dumps(result, default=_json_serial))
        metrics_table = _build_metrics_table(result)

        return fig, results_children, "", {"display": "block"}, store_result, metrics_table

    # --- Download callback ---
    @app.callback(
        Output("traj-download-component", "data"),
        Input("traj-download-btn", "n_clicks"),
        [State("traj-results-store", "data"),
         State("traj-k", "value"),
         State("traj-radius", "value")],
        prevent_initial_call=True,
    )
    def handle_download(n_clicks, result_data, k, radius):
        if not n_clicks or not result_data:
            raise PreventUpdate
        
        k = int(k) if k else 5
        radius = float(radius) if radius else None
        
        report_md = format_report(result_data, k=k, radius=radius)
        
        # Safely extract a slug for the filename – fallback to "synthetic" if no valid ID
        paper_id_raw = result_data["target"].get("id")
        if not paper_id_raw:
            paper_id_slug = "synthetic"
        else:
            paper_id_slug = str(paper_id_raw).split("/")[-1]
        filename = f"trajectory_report_{paper_id_slug}.md"
        
        return dict(content=report_md, filename=filename)


def _build_metrics_table(result: dict) -> dbc.Table:
    """Build a comparison table of trajectory and operator metrics."""
    metrics = result.get("trajectory_metrics", {})
    if not metrics:
        return html.Div("No metrics calculated.", className="text-muted")

    raw = metrics.get("embedding_raw", {})
    norm = metrics.get("embedding_normalized", {})
    proj = metrics.get("projection_2d", {})

    def fmt(val):
        if val is None or val == {}:
            return "N/A"
        return f"{val:.4f}"

    table_rows = [
        # Novelty
        html.Tr([
            html.Td(html.Strong("Novelty (||i - p||)")),
            html.Td(fmt(raw.get("novelty"))),
            html.Td(fmt(norm.get("novelty"))),
            html.Td(fmt(proj.get("novelty"))),
        ], style={"backgroundColor": "rgba(255, 193, 7, 0.05)"}),

        # Header for Step Sizes
        html.Tr([
            html.Td(html.Span("Operator Magnitudes (Step Sizes)", className="text-muted small fw-bold"), colSpan=4),
        ], style={"backgroundColor": "#1a1a2e"}),

        html.Tr([
            html.Td("  Problem → Method (||pm||)"),
            html.Td(fmt(raw.get("norm_pm"))),
            html.Td(fmt(norm.get("norm_pm"))),
            html.Td(fmt(proj.get("norm_pm"))),
        ]),
        html.Tr([
            html.Td("  Method → Finding (||mf||)"),
            html.Td(fmt(raw.get("norm_mf"))),
            html.Td(fmt(norm.get("norm_mf"))),
            html.Td(fmt(proj.get("norm_mf"))),
        ]),
        html.Tr([
            html.Td("  Finding → Interpretation (||fi||)"),
            html.Td(fmt(raw.get("norm_fi"))),
            html.Td(fmt(norm.get("norm_fi"))),
            html.Td(fmt(proj.get("norm_fi"))),
        ]),

        # Header for Alignments
        html.Tr([
            html.Td(html.Span("Operator Alignments (Cosine Similarity)", className="text-muted small fw-bold"), colSpan=4),
        ], style={"backgroundColor": "#1a1a2e"}),

        html.Tr([
            html.Td("  cos(pm, mf)"),
            html.Td(fmt(raw.get("cos_pm_mf"))),
            html.Td(fmt(norm.get("cos_pm_mf"))),
            html.Td(fmt(proj.get("cos_pm_mf"))),
        ]),
        html.Tr([
            html.Td("  cos(pm, fi)"),
            html.Td(fmt(raw.get("cos_pm_fi"))),
            html.Td(fmt(norm.get("cos_pm_fi"))),
            html.Td(fmt(proj.get("cos_pm_fi"))),
        ]),
        html.Tr([
            html.Td("  cos(mf, fi)"),
            html.Td(fmt(raw.get("cos_mf_fi"))),
            html.Td(fmt(norm.get("cos_mf_fi"))),
            html.Td(fmt(proj.get("cos_mf_fi"))),
        ]),
    ]

    return html.Div([
        dbc.Table(
            [
                html.Thead([
                    html.Tr([
                        html.Th("Metric Name"),
                        html.Th("Embedding (Raw)"),
                        html.Th("Embedding (Normalized)"),
                        html.Th("2D Projection"),
                    ])
                ]),
                html.Tbody(table_rows),
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="mb-2",
            style={"fontSize": "0.9rem"},
        ),
        html.Div([
            html.Small([
                html.Strong("Novelty (||i - p||): "),
                "Measures the distance between Interpretation and Problem. Higher indicates a more substantial departure (higher potential novelty/new problems opened), while lower suggests incremental confirmation.",
                html.Br(),
                html.Strong("Operator Magnitudes (Step Sizes): "),
                "The L2 distance moved during each stage of the paper's trajectory.",
                html.Br(),
                html.Strong("Operator Alignments: "),
                "Cosine similarity between successive movement directions. Values near 1 mean consecutive phases moved in the same direction, while negative values show opposing directions.",
            ], className="text-muted")
        ])
    ])
