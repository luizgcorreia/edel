"""Tests for Stage 5 Visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from edel.viz.vector_field import (
    plot_field_density,
    plot_field_magnitude,
    plot_vector_field,
)


@pytest.fixture
def field_df():
    """Create a mock vector field DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "cell_px": np.random.rand(25),
            "cell_py": np.random.rand(25),
            "vf_pm_x": np.random.rand(25),
            "vf_pm_y": np.random.rand(25),
            "vf_mf_x": np.random.rand(25),
            "vf_mf_y": np.random.rand(25),
            "vf_fi_x": np.random.rand(25),
            "vf_fi_y": np.random.rand(25),
            "mag_pm": np.random.rand(25),
            "count": np.random.randint(1, 100, 25),
        }
    )


def test_viz_vector_field_plots(field_df):
    """Verify that Stage 5 visualization functions run without errors."""
    plt.switch_backend("Agg")

    try:
        # Test basic vector field
        plot_vector_field(field_df, field_type="pm", min_count=1)

        # Test total field with color by magnitude
        plot_vector_field(
            field_df, 
            field_type="total", 
            color_by_mag=True, 
            min_count=0
        )

        # Test magnitude heatmap
        plot_field_magnitude(field_df, operator="pm")

        # Test density heatmap
        plot_field_density(field_df)

    except Exception as e:
        pytest.fail(f"Vector field visualization failed with error: {e}")
    finally:
        plt.close("all")
