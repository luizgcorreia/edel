"""Tests for Stage 8: Landscape."""

import numpy as np
import pandas as pd
import pytest
from edel.pipeline.landscape import run_landscape_stage


@pytest.fixture
def df_land():
    # 10 documents along a diagonal
    return pd.DataFrame(
        {
            "proj_problem_umap_x": np.linspace(0, 10, 10),
            "proj_problem_umap_y": np.linspace(0, 10, 10),
            "cited_by_count": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )


@pytest.fixture
def field_land():
    # 5 grid cells
    return pd.DataFrame(
        {
            "cell_px": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cell_py": [1.0, 2.0, 3.0, 4.0, 5.0],
            "vf_pm_x": [1.0, 0.0, -1.0, 0.0, 1.0],
            "vf_pm_y": [0.0, 1.0, 0.0, -1.0, 0.0],
        }
    )


def test_landscape_terrain(df_land):
    field = pd.DataFrame()
    config = {
        "dimensionality_reduction": {"method": "umap"},
        "landscape": {
            "metric": "cited_by_count",
            "grid": {
                "num_bins": 10,
                "sigma": 1.0
            }
        }
    }
    results = run_landscape_stage(df_land, field, config)
    assert "terrain" in results
    terrain = results["terrain"]
    # Check grid shapes
    assert terrain["x"].shape == (10, 10)
    assert terrain["z"].shape == (10, 10)
    # Z should be log10(cited_by_count + 1) in our implementation
    assert np.max(terrain["z"]) > 0
    assert terrain["metric"] == "cited_by_count"


def test_landscape_vf_smoothing(df_land, field_land):
    config = {
        "dimensionality_reduction": {"method": "umap"},
        "landscape": {"vf_resolution": 10, "vf_kernel_sigma": 0.5},
    }
    results = run_landscape_stage(df_land, field_land, config)
    assert "vector_field" in results
    vf = results["vector_field"]
    # Check grid shapes
    assert vf["x"].shape == (10, 10)
    assert vf["u"].shape == (10, 10)
    assert vf["v"].shape == (10, 10)


def test_landscape_empty_input():
    df = pd.DataFrame()
    field = pd.DataFrame()
    config = {"landscape": {}}
    results = run_landscape_stage(df, field, config)
    # Should handle empty gracefully
    assert results == {"terrain": {}}
