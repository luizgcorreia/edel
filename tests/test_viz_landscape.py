"""Tests for Stage 8 Visualizations."""

import numpy as np
import pandas as pd
import pytest
from edel.viz.landscape import plot_landscape_3d, plot_landscape_contour


@pytest.fixture
def df_land():
    """Create a mock landscape DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "proj_problem_umap_x": np.random.rand(10),
            "proj_problem_umap_y": np.random.rand(10),
            "cited_by_count": np.random.randint(0, 100, 10),
            "cluster_domain": [0, 1] * 5,
            "cluster_style": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "title": [f"Paper Title {i}" for i in range(10)],
            "publication_year": np.random.randint(2000, 2024, 10),
            "id": [f"doc_{i}" for i in range(10)],
        }
    )


@pytest.fixture
def landscape_results():
    """Mock results from landscape calculation stage."""
    np.random.seed(42)
    xi, yi = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    zi = np.random.rand(10, 10)
    return {
        "terrain": {
            "xi": xi,
            "yi": yi,
            "zi": zi,
            "metric": "cited_by_count",
            "log_scale": True,
        }
    }


@pytest.fixture
def field_df():
    """Mock vector field for flow overlay."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "cell_px": np.random.rand(5),
            "cell_py": np.random.rand(5),
            "vf_mf_x": np.random.rand(5),
            "vf_mf_y": np.random.rand(5),
            "vf_fi_x": np.random.rand(5),
            "vf_fi_y": np.random.rand(5),
        }
    )


def test_viz_landscape_plots(df_land, landscape_results, field_df):
    """Verify that Stage 8 visualization functions generate Plotly figures successfully."""
    
    # 1. Test 3D Surface
    fig3d = plot_landscape_3d(
        df_land, 
        landscape_results, 
        method="umap", 
        color_col="cluster_domain",
        symbol_col="cluster_style"
    )
    assert fig3d is not None
    assert len(fig3d.data) > 0  # Should have surface and scatter traces

    # 2. Test 2D Contour
    fig2d = plot_landscape_contour(
        df_land,
        landscape_results,
        field=field_df,
        method="umap",
        color_col="cluster_domain",
        show_flow=True,
        flow_type="discovery"
    )
    assert fig2d is not None
    assert len(fig2d.data) > 0
