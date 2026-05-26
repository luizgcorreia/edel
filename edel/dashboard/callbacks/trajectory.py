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
    """Build a rich neighbour card with expandable segments following D(Y|x) notation."""
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

    # Map aspects to Y_x mathematical notation
    aspect_math_map = {
        "problem": "Y_p (Problem)",
        "method": "Y_m (Method)",
        "finding": "Y_f (Finding)",
        "interpretation": "Y_i (Interpretation)"
    }

    # Segment rows — show current aspect segment first and prominently
    segment_items = [
        dbc.ListGroupItem([
            html.Strong(f"{aspect_math_map[asp]}: "),
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

def _build_trajectory_plot_2d(df: pd.DataFrame | None, result: dict, method: str) -> go.Figure:
    """Build the 2D scatter plot with a single p->i vector on the P space."""
    target = result["target"]
    aspects_data = result["aspects"]

    fig = go.Figure()

    if df is not None:
        # Background scatter (sample for speed)
        sample_size = min(2000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)

        x_col = f"proj_problem_{method}_x"
        y_col = f"proj_problem_{method}_y"

        if x_col in df_sample.columns and y_col in df_sample.columns:
            fig.add_trace(go.Scatter(
                x=df_sample[x_col],
                y=df_sample[y_col],
                mode="markers",
                marker=dict(size=3, color="rgba(100,120,200,0.25)"),
                name="All Papers",
                hoverinfo="skip",
            ))

    # Net displacement vector p -> i (Problem to Interpretation)
    p_coords = aspects_data.get("problem", {}).get("target_vec_2d")
    i_coords = aspects_data.get("interpretation", {}).get("target_vec_2d")

    if p_coords and i_coords:
        xs = [p_coords[0], i_coords[0]]
        ys = [p_coords[1], i_coords[1]]

        # Draw the displacement line
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers+text",
            line=dict(color="cyan", width=3, dash="dot"),
            marker=dict(size=[12, 10], color=["gold", "tomato"], symbol=["circle", "diamond"]),
            text=["p (Start)", "i (End)"],
            textposition="top center",
            textfont=dict(family="Arial Black", size=11, color="white"),
            name="Net Displacement (p -> i)",
        ))

        # Add single arrow annotation pointing from p to i
        fig.add_annotation(
            x=i_coords[0], y=i_coords[1],
            ax=p_coords[0], ay=p_coords[1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.5,
            arrowwidth=3, arrowcolor="cyan",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def _build_trajectory_simplex_plot_3d(
    df: pd.DataFrame | None,
    result: dict,
    method: str,
    selected_vertex: str
) -> go.Figure:
    """Build the 3D tetrahedron (3-simplex) plot bridging the 4 discourse spaces.

    Z values map to: Problem=0, Method=1, Finding=2, Interpretation=3.
    """
    aspects_data = result["aspects"]
    fig = go.Figure()

    z_map = {"problem": 0, "method": 1, "finding": 2, "interpretation": 3}
    
    vertex_xs, vertex_ys, vertex_zs, vertex_names, vertex_keys = [], [], [], [], []
    for asp in ASPECTS:
        coords = aspects_data.get(asp, {}).get("target_vec_2d")
        if coords:
            vertex_xs.append(coords[0])
            vertex_ys.append(coords[1])
            vertex_zs.append(z_map[asp])
            vertex_names.append(f"{asp.capitalize()}")
            vertex_keys.append(asp)

    if not vertex_xs:
        return fig

    # Draw sequential trajectory edges: P -> M -> F -> I (solid gold)
    fig.add_trace(go.Scatter3d(
        x=vertex_xs, y=vertex_ys, z=vertex_zs,
        mode="lines",
        line=dict(color="gold", width=6),
        name="Sequential Trajectory",
        hoverinfo="skip"
    ))

    # Draw cross-cutting simplex edges: P->F, P->I, M->I (dashed white)
    cross_xs, cross_ys, cross_zs = [], [], []
    connections = [("problem", "finding"), ("problem", "interpretation"), ("method", "interpretation")]
    for u, v in connections:
        if u in vertex_keys and v in vertex_keys:
            idx_u = vertex_keys.index(u)
            idx_v = vertex_keys.index(v)
            cross_xs.extend([vertex_xs[idx_u], vertex_xs[idx_v], None])
            cross_ys.extend([vertex_ys[idx_u], vertex_ys[idx_v], None])
            cross_zs.extend([vertex_zs[idx_u], vertex_zs[idx_v], None])

    fig.add_trace(go.Scatter3d(
        x=cross_xs, y=cross_ys, z=cross_zs,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.4)", width=3, dash="dash"),
        name="Simplex Structure",
        hoverinfo="skip"
    ))

    # Draw markers for vertices with custom labels and selection highlight
    marker_sizes = []
    marker_colors = []
    marker_symbols = []
    for asp in ASPECTS:
        if asp == selected_vertex:
            marker_sizes.append(15)
            marker_colors.append("orange")
            marker_symbols.append("circle")
        else:
            marker_sizes.append(10)
            marker_colors.append("white")
            marker_symbols.append("circle")

    fig.add_trace(go.Scatter3d(
        x=vertex_xs, y=vertex_ys, z=vertex_zs,
        mode="markers+text",
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            symbol=marker_symbols,
            line=dict(color="black", width=2)
        ),
        text=vertex_names,
        textposition="top center",
        textfont=dict(color="white", size=12, family="Arial Black"),
        customdata=vertex_keys,
        name="Spaces",
        hovertemplate="<b>%{text} Space</b><br>Click to inspect<extra></extra>"
    ))

    # Overlay neighbors' trajectories across all four spaces (D(Y|x))
    selected_asp_data = aspects_data.get(selected_vertex, {})
    neighbors = selected_asp_data.get("neighbors", [])
    
    neighbor_colors = [
        "rgba(255, 99, 71, 0.7)",   # Tomato
        "rgba(30, 144, 255, 0.7)",  # DodgerBlue
        "rgba(46, 139, 87, 0.7)",   # SeaGreen
        "rgba(218, 112, 214, 0.7)", # Orchid
        "rgba(255, 215, 0, 0.7)",   # Gold
    ]

    for idx_n, n in enumerate(neighbors):
        if df is None:
            continue
        n_data = df[df["id"] == n["id"]]
        if n_data.empty:
            continue
        n_row = n_data.iloc[0]

        # Extract coordinates for each of the four spaces
        n_xs, n_ys, n_zs, n_hovertexts = [], [], [], []
        valid = True
        for asp in ASPECTS:
            asp_x_col = f"proj_{asp}_{method}_x"
            asp_y_col = f"proj_{asp}_{method}_y"
            if asp_x_col in df.columns and pd.notna(n_row.get(asp_x_col)):
                n_xs.append(float(n_row[asp_x_col]))
                n_ys.append(float(n_row[asp_y_col]))
                n_zs.append(z_map[asp])
                n_hovertexts.append(
                    f"Neighbor {idx_n+1}: {n.get('title', 'Unknown')[:50]}...<br>"
                    f"Discourse Space: {asp.capitalize()}<br>"
                    f"Distance to target in {selected_vertex.capitalize()}: {n['distance']:.4f}"
                )
            else:
                valid = False
                break

        if valid and len(n_xs) == 4:
            # Consistent color for this neighbor
            color = neighbor_colors[idx_n % len(neighbor_colors)]
            
            # Add line trace for the neighbor simplex trajectory
            fig.add_trace(go.Scatter3d(
                x=n_xs, y=n_ys, z=n_zs,
                mode="lines+markers",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=4, symbol="diamond", color=color),
                name=f"Neighbor: {n.get('title', 'Unknown')[:15]}...",
                hovertemplate="%{text}<extra></extra>",
                text=n_hovertexts,
                legendgroup=f"neighbor_{idx_n}",
            ))

    # 3D layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        scene=dict(
            xaxis=dict(showgrid=False, showbackground=False, showaxeslabels=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showbackground=False, showaxeslabels=False, showticklabels=False, title=""),
            zaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["P (Problem)", "M (Method)", "F (Finding)", "I (Interpretation)"],
                backgroundcolor="rgba(0,0,0,0.2)",
                gridcolor="rgba(255,255,255,0.1)",
                title=""
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2)
            )
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _build_neighborhood_results_html(df: pd.DataFrame | None, result: dict, selected_vertex: str) -> list:
    """Build neighborhood results section formatted according to D(Y|x) notation."""
    aspect_result = result["aspects"].get(selected_vertex, {})
    target = result["target"]
    neighbors = aspect_result.get("neighbors", [])
    error = aspect_result.get("error")

    target_header = dbc.Card([
        dbc.CardBody([
            html.H6([
                html.Strong(target.get("title", "(Synthetic Paper)")),
                dbc.Badge(
                    f"{target.get('publication_year', '')} · {target.get('cited_by_count', '')} citations",
                    color="dark", className="ms-2"
                ),
            ], className="mb-1"),
            html.Div([
                html.Span("Conditioning vertex: ", className="text-muted"),
                html.Strong(f"x = {selected_vertex.capitalize()}", className="text-warning")
            ], className="small mt-1")
        ])
    ], color="dark", className="mb-3")

    body = [target_header]

    if error:
        body.append(dbc.Alert(f"Conditioning Space Error: {error}", color="warning"))
    elif not neighbors:
        body.append(html.P(f"No neighbors found for x = {selected_vertex}.", className="text-muted"))
    else:
        body.append(html.H5(f"Neighbor distribution D(Y | x = {selected_vertex.capitalize()})", className="mb-3 small text-muted text-uppercase"))
        body.extend([_neighbor_card(n, i + 1, selected_vertex) for i, n in enumerate(neighbors)])

    return body


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
        [Output("traj-status-msg", "children"),
         Output("traj-download-btn", "style"),
         Output("traj-results-store", "data"),
         Output("traj-metrics-container", "children"),
         Output("traj-selected-vertex", "data")],
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
            return f"Failed to load dataset: {e}", {"display": "none"}, None, dash.no_update, dash.no_update

        try:
            if input_mode == "work_id":
                if not paper_id or not paper_id.strip():
                    return "Please enter a Work ID.", {"display": "none"}, None, dash.no_update, dash.no_update
                paper_id = paper_id.strip()
                if paper_id not in df["id"].values:
                    return f"Work ID not found: {paper_id}", {"display": "none"}, None, dash.no_update, dash.no_update

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
                    return "Please enter at least one text segment.", {"display": "none"}, None, dash.no_update, dash.no_update

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
            return f"Analysis error: {e}", {"display": "none"}, None, dash.no_update, dash.no_update

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

        return "", {"display": "block"}, store_result, metrics_table, "problem"

    # --- 3D Graph Click to select vertex ---
    @app.callback(
        Output("traj-selected-vertex", "data"),
        Input("traj-plot-3d", "clickData"),
        State("traj-selected-vertex", "data"),
        prevent_initial_call=True
    )
    def select_vertex_from_click(click_data, current_vertex):
        if not click_data:
            raise PreventUpdate
        try:
            point = click_data["points"][0]
            clicked_vertex = point.get("customdata")
            if clicked_vertex in ["problem", "method", "finding", "interpretation"]:
                return clicked_vertex
        except Exception as e:
            logger.warning(f"Error extracting clicked vertex: {e}")
        raise PreventUpdate

    # --- Render trajectory plots and neighborhood results ---
    @app.callback(
        [Output("traj-plot", "figure"),
         Output("traj-plot-3d", "figure"),
         Output("traj-selected-vertex-label", "children"),
         Output("traj-selected-segment-text", "children"),
         Output("traj-selected-vertex-container", "style"),
         Output("traj-results-container", "children")],
        [Input("traj-results-store", "data"),
         Input("traj-selected-vertex", "data")],
        [State("traj-experiment-select", "value"),
         State("traj-method-select", "value")],
    )
    def render_trajectory_results(result, selected_vertex, experiment_name, method):
        if not result or not experiment_name:
            empty_fig = go.Figure(layout={
                "template": "plotly_dark",
                "paper_bgcolor": "#1a1a2e",
                "plot_bgcolor": "#16213e",
                "height": 380,
                "xaxis": {"visible": False},
                "yaxis": {"visible": False}
            })
            return (
                empty_fig, empty_fig, "Problem",
                "Run an analysis to inspect trajectory.",
                {"display": "none"},
                html.P("Run an analysis to see results.", className="text-muted")
            )

        try:
            df = _load_dr_df(experiment_name, base_path)
        except Exception:
            df = None

        selected_vertex = (selected_vertex or "problem").lower()
        target = result.get("target", {})
        
        segment_text = target.get(selected_vertex, "(empty)")
        vertex_label = selected_vertex.upper()
        
        fig_2d = _build_trajectory_plot_2d(df, result, method)
        fig_3d = _build_trajectory_simplex_plot_3d(df, result, method, selected_vertex)
        results_children = _build_neighborhood_results_html(df, result, selected_vertex)

        return (
            fig_2d,
            fig_3d,
            vertex_label,
            segment_text,
            {"display": "block"},
            results_children
        )

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

    # --- Fullscreen toggle for 2D plot ---
    @app.callback(
        [Output("traj-plot-2d-card", "className"),
         Output("traj-plot", "style"),
         Output("traj-plot-fullscreen-btn", "children")],
        Input("traj-plot-fullscreen-btn", "n_clicks"),
        State("traj-plot-2d-card", "className"),
        prevent_initial_call=True
    )
    def toggle_fullscreen_2d(n_clicks, current_class):
        if not current_class:
            current_class = ""
        if "fullscreen-graph" in current_class:
            return "mb-3", {"height": "380px"}, "⛶ Fullscreen"
        else:
            return "fullscreen-graph", {"height": "calc(100vh - 80px)"}, "🗖 Exit Fullscreen"

    # --- Fullscreen toggle for 3D plot ---
    @app.callback(
        [Output("traj-plot-3d-card", "className"),
         Output("traj-plot-3d", "style"),
         Output("traj-plot-3d-fullscreen-btn", "children")],
        Input("traj-plot-3d-fullscreen-btn", "n_clicks"),
        State("traj-plot-3d-card", "className"),
        prevent_initial_call=True
    )
    def toggle_fullscreen_3d(n_clicks, current_class):
        if not current_class:
            current_class = ""
        if "fullscreen-graph" in current_class:
            return "mb-3", {"height": "380px"}, "⛶ Fullscreen"
        else:
            return "fullscreen-graph", {"height": "calc(100vh - 80px)"}, "🗖 Exit Fullscreen"


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
        # Cycle Closure & Net Epistemic Displacement
        html.Tr([
            html.Td(html.Strong("Cycle Closure (||i - p||)")),
            html.Td(fmt(raw.get("cycle_closure_norm"))),
            html.Td(fmt(norm.get("cycle_closure_norm"))),
            html.Td(fmt(proj.get("cycle_closure_norm"))),
        ], style={"backgroundColor": "rgba(255, 193, 7, 0.05)"}),

        html.Tr([
            html.Td(html.Strong("Net Epistemic Displacement (||p̄ - p||)")),
            html.Td(fmt(raw.get("net_epistemic_displacement_norm"))),
            html.Td(fmt(norm.get("net_epistemic_displacement_norm"))),
            html.Td(fmt(proj.get("net_epistemic_displacement_norm"))),
        ], style={"backgroundColor": "rgba(40, 167, 69, 0.05)"}),

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
                html.Strong("Cycle Closure (||i - p||): "),
                "Measures the distance between the final Interpretation and the starting Problem. Higher values indicate a larger gap (open-ended inquiry/high novelty), while lower values suggest a tightly closed cycle.",
                html.Br(),
                html.Strong("Net Epistemic Displacement (||p̄ - p||): "),
                "Measures the distance from the starting problem to the average problem opened by similar interpretations, representing the expected future intellectual direction.",
                html.Br(),
                html.Strong("Operator Magnitudes (Step Sizes): "),
                "The L2 distance moved during each stage of the paper's trajectory.",
                html.Br(),
                html.Strong("Operator Alignments: "),
                "Cosine similarity between successive movement directions. Values near 1 mean consecutive phases moved in the same direction, while negative values show opposing directions.",
            ], className="text-muted")
        ])
    ])
