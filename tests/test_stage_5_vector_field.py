"""Tests for Stage 5: Vector Field."""

import numpy as np
import pandas as pd
import pytest
from edel.pipeline.vector_field import run_vector_field_stage


@pytest.fixture
def df_projected():
    # 3 docs in 2D space
    # Doc 1: (0,0) -> (1,1) -> (2,2) -> (3,3)
    # Doc 2: (10,10) -> (9,9) -> (8,8) -> (7,7)
    # Doc 3: (0.1, 0.1) -> (1.1, 1.1) -> (2.1, 2.1) -> (3.1, 3.1) (Close to Doc 1)

    return pd.DataFrame(
        [
            {
                "proj_problem_umap_x": 0.0,
                "proj_problem_umap_y": 0.0,
                "proj_method_umap_x": 1.0,
                "proj_method_umap_y": 1.0,
                "proj_finding_umap_x": 2.0,
                "proj_finding_umap_y": 2.0,
                "proj_interpretation_umap_x": 3.0,
                "proj_interpretation_umap_y": 3.0,
            },
            {
                "proj_problem_umap_x": 10.0,
                "proj_problem_umap_y": 10.0,
                "proj_method_umap_x": 9.0,
                "proj_method_umap_y": 9.0,
                "proj_finding_umap_x": 8.0,
                "proj_finding_umap_y": 8.0,
                "proj_interpretation_umap_x": 7.0,
                "proj_interpretation_umap_y": 7.0,
            },
            {
                "proj_problem_umap_x": 0.1,
                "proj_problem_umap_y": 0.1,
                "proj_method_umap_x": 1.1,
                "proj_method_umap_y": 1.1,
                "proj_finding_umap_x": 2.1,
                "proj_finding_umap_y": 2.1,
                "proj_interpretation_umap_x": 3.1,
                "proj_interpretation_umap_y": 3.1,
            },
        ]
    )


def test_vector_field_computation(df_projected):
    config = {
        "dimensionality_reduction": {"method": "umap"},
        "vector_field": {"grid_size": 10, "min_count": 1, "compute_magnitude": True},
    }
    field = run_vector_field_stage(df_projected, config)

    assert isinstance(field, pd.DataFrame)
    # 2 cells should be present (one for Doc 1 & 3, one for Doc 2)
    assert len(field) == 2

    # Cell with Doc 1 and 3 should have count 2
    # They are at (0,0) and (0.1, 0.1) in a grid of 10x10 over [0, 10]
    # dx = (10 - 0) / 10 = 1.0
    # cell_ix = floor(0 / 1) = 0, floor(0.1 / 1) = 0
    # So they indeed fall into the same cell (0,0)
    cell_1_3 = field[field["count"] == 2].iloc[0]
    # Average p->m vector: ((1-0) + (1.1-0.1)) / 2 = 1.0
    assert pytest.approx(cell_1_3["vf_pm_x"]) == 1.0
    assert pytest.approx(cell_1_3["vf_pm_y"]) == 1.0

    # Magnitude check
    assert "mag_pm" in field.columns
    assert pytest.approx(cell_1_3["mag_pm"]) == np.sqrt(2)

    # Cell with Doc 2
    # (10, 10) in [0, 10] grid with size 10
    # cell_ix = floor((10-0)/1) = 10 -> clipped to 9
    cell_2 = field[field["count"] == 1].iloc[0]
    # Average p->m vector: (9-10) = -1.0
    assert pytest.approx(cell_2["vf_pm_x"]) == -1.0
    assert pytest.approx(cell_2["vf_pm_y"]) == -1.0


def test_vector_field_empty_input():
    df = pd.DataFrame()
    config = {"vector_field": {"min_count": 1}}
    field = run_vector_field_stage(df, config)
    assert field.empty
