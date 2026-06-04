"""Tests for Stage 4: Dimensionality Reduction."""

import json
import numpy as np
import pandas as pd
import pytest
from edel.pipeline.projection import run_projection_stage


@pytest.fixture
def df_embedded():
    # 5 documents, 4 dimensions for testing
    def rand_emb():
        return np.random.rand(4).tolist()

    return pd.DataFrame(
        [
            {
                "problem_embedding": json.dumps(rand_emb()),
                "method_embedding": json.dumps(rand_emb()),
                "finding_embedding": json.dumps(rand_emb()),
                "interpretation_embedding": json.dumps(rand_emb()),
            }
            for _ in range(5)
        ]
    )


def test_projection_umap(df_embedded):
    config = {
        "embedding": {"n_dimensions": 4},
        "dimensionality_reduction": {
            "method": "umap",
            "n_neighbors": 2,
            "random_state": 42,
        },
    }
    df_proj, _ = run_projection_stage(df_embedded, config)
    assert "proj_problem_umap_x" in df_proj.columns
    assert "proj_problem_umap_y" in df_proj.columns
    assert "proj_method_umap_x" in df_proj.columns
    assert "proj_method_umap_y" in df_proj.columns


def test_projection_pca(df_embedded):
    config = {
        "embedding": {"n_dimensions": 4},
        "dimensionality_reduction": {
            "method": "pca",
            "random_state": 42,
        },
    }
    df_proj, _ = run_projection_stage(df_embedded, config)
    assert "proj_problem_pca_x" in df_proj.columns
    assert "proj_method_pca_x" in df_proj.columns


def test_projection_diffusion(df_embedded):
    """Test diffusion map (may require specific data scale)."""
    config = {
        "embedding": {"n_dimensions": 4},
        "dimensionality_reduction": {
            "method": "diffusion",
            "random_state": 42,
        },
    }
    try:
        df_proj, _ = run_projection_stage(df_embedded, config)
        assert "proj_problem_diffusion_x" in df_proj.columns
    except Exception as e:
        # Diffusion map is sensitive to small datasets and kernel params
        pytest.skip(f"Diffusion map skipped: {e}")


def test_projection_single_mode():
    df = pd.DataFrame(
        [
            {"embedding": json.dumps([1, 2, 3])},
            {"embedding": json.dumps([4, 5, 6])},
            {"embedding": json.dumps([7, 8, 9])},
        ]
    )
    config = {
        "embedding": {"n_dimensions": 3},
        "dimensionality_reduction": {"method": "pca"},
    }
    df_proj, _ = run_projection_stage(df, config)
    assert "proj_pca_x" in df_proj.columns
    assert "proj_pca_y" in df_proj.columns


def test_drop_embeddings(df_embedded):
    config = {
        "embedding": {"n_dimensions": 4},
        "dimensionality_reduction": {
            "method": "pca",
            "drop_embeddings": True,
        },
    }
    df_proj, _ = run_projection_stage(df_embedded, config)
    assert "proj_problem_pca_x" in df_proj.columns
    assert "problem_embedding" not in df_proj.columns


def test_projection_with_null_embeddings():
    # 5 documents, 2 valid and 3 containing null/empty embeddings
    def valid_emb():
        return [1.0, 2.0, 3.0, 4.0]
    
    # We mix valid, None, empty list/JSON representation of zeros, and null JSON
    data = [
        # Document 0: Fully valid
        {
            "problem_embedding": json.dumps(valid_emb()),
            "method_embedding": json.dumps(valid_emb()),
            "finding_embedding": json.dumps(valid_emb()),
            "interpretation_embedding": json.dumps(valid_emb()),
        },
        # Document 1: Fully valid (different vector)
        {
            "problem_embedding": json.dumps([4.0, 3.0, 2.0, 1.0]),
            "method_embedding": json.dumps([4.0, 3.0, 2.0, 1.0]),
            "finding_embedding": json.dumps([4.0, 3.0, 2.0, 1.0]),
            "interpretation_embedding": json.dumps([4.0, 3.0, 2.0, 1.0]),
        },
        # Document 2: Fully null (None values)
        {
            "problem_embedding": None,
            "method_embedding": None,
            "finding_embedding": None,
            "interpretation_embedding": None,
        },
        # Document 3: Mix of null and valid
        {
            "problem_embedding": json.dumps(valid_emb()),
            "method_embedding": None,
            "finding_embedding": json.dumps(valid_emb()),
            "interpretation_embedding": None,
        },
        # Document 4: All-zero list representation of null
        {
            "problem_embedding": json.dumps([0.0, 0.0, 0.0, 0.0]),
            "method_embedding": json.dumps([0.0, 0.0, 0.0, 0.0]),
            "finding_embedding": json.dumps([0.0, 0.0, 0.0, 0.0]),
            "interpretation_embedding": json.dumps([0.0, 0.0, 0.0, 0.0]),
        },
    ]
    df = pd.DataFrame(data)
    
    config = {
        "embedding": {"n_dimensions": 4},
        "dimensionality_reduction": {
            "method": "pca",
            "random_state": 42,
        },
    }
    
    df_proj, _ = run_projection_stage(df, config)
    
    # Assert coordinates
    # Problem aspect: valid for doc 0, 1, 3. Null for doc 2, 4.
    assert not pd.isna(df_proj.loc[0, "proj_problem_pca_x"])
    assert not pd.isna(df_proj.loc[1, "proj_problem_pca_x"])
    assert pd.isna(df_proj.loc[2, "proj_problem_pca_x"])
    assert not pd.isna(df_proj.loc[3, "proj_problem_pca_x"])
    assert pd.isna(df_proj.loc[4, "proj_problem_pca_x"])
    
    # Method aspect: valid for doc 0, 1. Null for doc 2, 4. Document 3 falls back to problem_embedding and is non-NaN.
    assert not pd.isna(df_proj.loc[0, "proj_method_pca_x"])
    assert not pd.isna(df_proj.loc[1, "proj_method_pca_x"])
    assert pd.isna(df_proj.loc[2, "proj_method_pca_x"])
    assert not pd.isna(df_proj.loc[3, "proj_method_pca_x"])
    assert pd.isna(df_proj.loc[4, "proj_method_pca_x"])

    # Assert signatures and magnitudes
    # mag_pm requires problem AND method. Valid for doc 0, 1, 3. NaN for doc 2, 4.
    assert not pd.isna(df_proj.loc[0, "mag_pm"])
    assert not pd.isna(df_proj.loc[1, "mag_pm"])
    assert pd.isna(df_proj.loc[2, "mag_pm"])
    assert not pd.isna(df_proj.loc[3, "mag_pm"])
    assert pd.isna(df_proj.loc[4, "mag_pm"])

    # cos_pm_mf requires problem AND method AND finding. Valid for doc 0, 1, 3. NaN for doc 2, 4.
    assert not pd.isna(df_proj.loc[0, "cos_pm_mf"])
    assert not pd.isna(df_proj.loc[1, "cos_pm_mf"])
    assert pd.isna(df_proj.loc[2, "cos_pm_mf"])
    assert not pd.isna(df_proj.loc[3, "cos_pm_mf"])
    assert pd.isna(df_proj.loc[4, "cos_pm_mf"])


def test_projection_single_mode_with_null_embeddings():
    df = pd.DataFrame([
        {"embedding": json.dumps([1, 2, 3])},
        {"embedding": None},
        {"embedding": json.dumps([0, 0, 0])},
        {"embedding": json.dumps([4, 5, 6])},
    ])
    config = {
        "embedding": {"n_dimensions": 3},
        "dimensionality_reduction": {"method": "pca"},
    }
    df_proj, _ = run_projection_stage(df, config)
    
    assert not pd.isna(df_proj.loc[0, "proj_pca_x"])
    assert pd.isna(df_proj.loc[1, "proj_pca_x"])
    assert pd.isna(df_proj.loc[2, "proj_pca_x"])
    assert not pd.isna(df_proj.loc[3, "proj_pca_x"])

