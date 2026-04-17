"""Vector-field visualization helpers."""


def add_vector_field_annotations(fig, vectors, width: float = 1.0):
    """Add arrow annotations for a precomputed vector field to a Plotly figure."""
    if vectors is None:
        return fig

    for x, y, dx, dy in zip(vectors["x"], vectors["y"], vectors["dx"], vectors["dy"]):
        fig.add_annotation(
            x=x + dx,
            y=y + dy,
            ax=x,
            ay=y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=1,
            arrowsize=0.6,
            arrowwidth=width,
            arrowcolor="black",
            opacity=0.8,
        )

    return fig
