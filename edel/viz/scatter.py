"""2D scatter visualization helpers."""

import plotly.express as px


def make_scatter(df, X, Y, color_col=None, style_col=None, opacity: float = 0.3):
    """Create a 2D scatter figure from precomputed XY coordinates and labels."""
    fig = px.scatter(
        df,
        x=X,
        y=Y,
        color=color_col,
        symbol=style_col,
        opacity=opacity,
        color_discrete_sequence=px.colors.qualitative.Set1,
        symbol_sequence=["circle", "diamond", "square", "cross", "x", "triangle-up"],
    )

    for trace in fig.data:
        trace.marker.size = 6

    return fig
