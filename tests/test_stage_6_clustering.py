"""Tests for Stage 6: Clustering."""

import json
import numpy as np
import pandas as pd
import pytest
from edel.pipeline.clustering import run_clustering_stage


@pytest.fixture
def df_results():
    # 10 documents, 2D projections
    np.random.seed(42)
    return pd.DataFrame(
        {
            "proj_problem_umap_x": np.random.rand(10),
            "proj_problem_umap_y": np.random.rand(10),
            "problem_embedding": [
                json.dumps(np.random.rand(4).tolist()) for _ in range(10)
            ],
            "method_embedding": [
                json.dumps(np.random.rand(4).tolist()) for _ in range(10)
            ],
            "finding_embedding": [
                json.dumps(np.random.rand(4).tolist()) for _ in range(10)
            ],
            "interpretation_embedding": [
                json.dumps(np.random.rand(4).tolist()) for _ in range(10)
            ],
        }
    )


@pytest.fixture
def field_results():
    # 5 grid cells
    np.random.seed(42)
    return pd.DataFrame(
        {
            "vf_pm_x": np.random.rand(5),
            "vf_pm_y": np.random.rand(5),
            "mag_pm": np.random.rand(5),
        }
    )


def test_clustering_kmeans_proj(df_results, field_results):
    config = {
        "embedding": {"n_dimensions": 4},
        "clustering": {
            "domains": {
                "source": "proj_p",
                "algorithm": "kmeans",
                "params": {"n_clusters": 2, "random_state": 42, "n_init": 10},
            }
        },
    }
    df_c, field_c = run_clustering_stage(df_results, field_results, config)
    assert "cluster_domains" in df_c.columns
    assert len(df_c["cluster_domains"].unique()) <= 2


def test_clustering_hdbscan_emb(df_results, field_results):
    config = {
        "embedding": {"n_dimensions": 4},
        "clustering": {
            "hdb": {
                "source": "emb_p",
                "algorithm": "hdbscan",
                "params": {"min_cluster_size": 2},
            }
        },
    }
    df_c, field_c = run_clustering_stage(df_results, field_results, config)
    assert "cluster_hdb" in df_c.columns


def test_clustering_features(df_results, field_results):
    config = {
        "embedding": {"n_dimensions": 4},
        "clustering": {
            "styles": {
                "source": "features",
                "algorithm": "kmeans",
                "params": {"n_clusters": 3, "n_init": 10},
            }
        },
    }
    df_c, field_c = run_clustering_stage(df_results, field_results, config)
    assert "cluster_styles" in df_c.columns


def test_clustering_field(df_results, field_results):
    config = {
        "clustering": {
            "flow": {
                "source": "field",
                "algorithm": "kmeans",
                "params": {"n_clusters": 2, "n_init": 10},
            }
        },
    }
    df_c, field_c = run_clustering_stage(df_results, field_results, config)
    assert "cluster_flow" in field_c.columns
