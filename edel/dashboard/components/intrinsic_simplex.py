"""Reusable, distance-preserving discourse-simplex visualisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from edel.analysis.trajectory import ASPECTS, parse_embedding_vector


ASPECT_COLORS = {
    "problem": "#4A90E2", "method": "#F5A623",
    "finding": "#7ED321", "interpretation": "#BD10E0",
}


def intrinsic_simplex_coordinates(row: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Return canonical 3D coordinates and all pairwise embedding distances."""
    vectors = [parse_embedding_vector(row.get(f"{aspect}_embedding")) for aspect in ASPECTS]
    if any(vector is None for vector in vectors):
        return None
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        return None

    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    n = len(ASPECTS)
    centre = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centre @ (distances ** 2) @ centre
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    coordinates = vectors[:, order][:, :3] * np.sqrt(np.clip(values[order][:3], 0.0, None))

    # Stable display orientation: P at origin, P→M on x, then F and I.
    relative = coordinates - coordinates[0]
    basis: list[np.ndarray] = []
    tolerance = max(float(np.max(distances)), 1.0) * 1e-10
    for vector in relative[1:]:
        residual = vector.copy()
        for axis in basis:
            residual -= np.dot(residual, axis) * axis
        norm = np.linalg.norm(residual)
        if norm > tolerance:
            basis.append(residual / norm)
    oriented = np.zeros((n, 3))
    for index, axis in enumerate(basis):
        oriented[:, index] = relative @ axis
    return oriented, distances


def _edge_segments(coordinates: np.ndarray, connections: list[tuple[int, int]], inset: float = 0.025) -> tuple[list[float | None], list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for first, second in connections:
        start, end = coordinates[first], coordinates[second]
        delta = end - start
        start, end = start + inset * delta, end - inset * delta
        xs.extend([start[0], end[0], None])
        ys.extend([start[1], end[1], None])
        zs.extend([start[2], end[2], None])
    return xs, ys, zs


def build_intrinsic_simplex_figure(target_row: pd.Series | None, neighbor_rows: list[tuple[pd.Series, dict]], selected_aspect: str) -> go.Figure:
    """Render target and selected-aspect neighbours as unnormalised tetrahedra."""
    fig = go.Figure()
    if target_row is None:
        fig.add_annotation(text="Intrinsic simplex requires a stored paper embedding.", showarrow=False)
        return fig
    target = intrinsic_simplex_coordinates(target_row)
    if target is None:
        fig.add_annotation(text="Aspect embeddings unavailable for this paper.", showarrow=False)
        return fig

    coordinates, distances = target
    sequential = _edge_segments(coordinates, [(0, 1), (1, 2), (2, 3)])
    cross = _edge_segments(coordinates, [(0, 2), (0, 3), (1, 3)])
    fig.add_trace(go.Scatter3d(x=sequential[0], y=sequential[1], z=sequential[2], mode="lines", line=dict(color="gold", width=6), name="Target trajectory", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=cross[0], y=cross[1], z=cross[2], mode="lines", line=dict(color="rgba(180,180,180,0.8)", width=3, dash="dash"), name="Target simplex edges", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2], mode="markers+text",
        marker=dict(size=[15 if aspect == selected_aspect else 11 for aspect in ASPECTS], color=[ASPECT_COLORS[aspect] for aspect in ASPECTS], line=dict(color="black", width=1)),
        text=[aspect.capitalize() for aspect in ASPECTS], textposition="top center", customdata=ASPECTS,
        name="Target aspects", hovertemplate="<b>%{text}</b><br>Click to inspect neighbours<extra></extra>",
    ))

    for index, (neighbor_row, neighbor) in enumerate(neighbor_rows):
        simplex = intrinsic_simplex_coordinates(neighbor_row)
        if simplex is None:
            continue
        neighbor_coords, _ = simplex
        edges = _edge_segments(neighbor_coords, [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)], inset=0.01)
        color = ["#FF6347", "#1E90FF", "#2E8B57", "#DA70D6", "#FFD700"][index % 5]
        fig.add_trace(go.Scatter3d(
            x=edges[0], y=edges[1], z=edges[2], mode="lines", line=dict(color=color, width=2, dash="dash"),
            name=f"Neighbour {index + 1}: {str(neighbor.get('title', 'Unknown'))[:22]}",
            hovertemplate=f"<b>Neighbour {index + 1}</b><br>Distance: {neighbor.get('distance', 0):.4f}<extra></extra>",
            legendgroup=f"neighbor-{index}",
        ))
        fig.add_trace(go.Scatter3d(x=neighbor_coords[:, 0], y=neighbor_coords[:, 1], z=neighbor_coords[:, 2], mode="markers", marker=dict(size=4, color=color, symbol="diamond"), showlegend=False, hoverinfo="skip", legendgroup=f"neighbor-{index}"))

    padding = max(float(np.max(distances)) * 0.08, 1e-6)
    ranges = [[float(coordinates[:, axis].min()) - padding, float(coordinates[:, axis].max()) + padding] for axis in range(3)]
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e", margin=dict(l=0, r=0, t=30, b=0),
        title="Intrinsic Discourse Simplex (embedding distances)",
        scene=dict(
            xaxis=dict(title="Intrinsic axis 1 (P→M)", range=ranges[0], showbackground=False),
            yaxis=dict(title="Intrinsic axis 2", range=ranges[1], showbackground=False),
            zaxis=dict(title="Intrinsic axis 3", range=ranges[2], showbackground=False),
            aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return fig
