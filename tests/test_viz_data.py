"""Tests for Stage 1 Visualizations."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from edel.viz.data import (
    plot_abstract_length_dist,
    plot_citation_dist,
    plot_publication_year_dist,
)


def test_viz_data_plots():
    """Verify that Stage 1 visualization functions run without errors."""
    df = pd.DataFrame(
        {
            "abstract_text": ["word " * 10, "word " * 20, "word " * 30],
            "publication_year": [2020, 2021, 2022],
            "cited_by_count": [5, 10, 15],
        }
    )

    # Use a non-interactive backend for headless testing
    plt.switch_backend("Agg")

    try:
        plot_abstract_length_dist(df)
        plot_publication_year_dist(df)
        plot_citation_dist(df)
    except Exception as e:
        pytest.fail(f"Visualization failed with error: {e}")
    finally:
        plt.close("all")
