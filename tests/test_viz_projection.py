"""Tests for Stage 4 Visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from edel.viz.projection import (
    plot_movement_magnitudes,
    plot_projection_2d,
    plot_transition_signatures,
)


@pytest.fixture
def df_proj():
    """Create a mock projection DataFrame."""
    np.random.seed(42)
    data = {
        "proj_problem_umap_x": np.random.rand(10),
        "proj_problem_umap_y": np.random.rand(10),
        "proj_method_umap_x": np.random.rand(10),
        "proj_method_umap_y": np.random.rand(10),
        "proj_finding_umap_x": np.random.rand(10),
        "proj_finding_umap_y": np.random.rand(10),
        "proj_interpretation_umap_x": np.random.rand(10),
        "proj_interpretation_umap_y": np.random.rand(10),
        "cos_pm_mf": np.random.rand(10),
        "cos_pm_fi": np.random.rand(10),
        "cos_mf_fi": np.random.rand(10),
        "mag_pm": np.random.rand(10),
        "mag_mf": np.random.rand(10),
        "mag_fi": np.random.rand(10),
        "cluster_id": np.random.randint(0, 3, 10),
    }
    return pd.DataFrame(data)


def test_viz_projection_plots(df_proj):
    """Verify that Stage 4 visualization functions run without errors."""
    plt.switch_backend("Agg")

    try:
        # Test basic scatter
        plot_projection_2d(df_proj, method="umap")
        
        # Test with color and arrows
        plot_projection_2d(
            df_proj, 
            method="umap", 
            color_col="cluster_id", 
            draw_arrows=True, 
            arrow_step=2
        )
        
        # Test signatures
        plot_transition_signatures(df_proj)
        
        # Test magnitudes
        plot_movement_magnitudes(df_proj)
        
    except Exception as e:
        pytest.fail(f"Projection visualization failed with error: {e}")
    finally:
        plt.close("all")
