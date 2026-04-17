"""Contour map visualization helpers."""

import plotly.graph_objects as go


def make_contour_figure(
    xi,
    yi,
    grid_smooth,
    z_label,
    title,
    x_label,
    y_label,
):
    """Build a contour figure from precomputed grid artifacts."""
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            z=grid_smooth,
            x=xi[0],
            y=yi[:, 0],
            colorscale="Viridis",
            contours={"showlabels": True, "labelfont": {"size": 10, "color": "white"}},
            colorbar={"title": z_label},
            opacity=0.9,
            name="Contours",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=1000,
        height=900,
        margin={"l": 50, "r": 50, "b": 100, "t": 80},
    )

    return fig
