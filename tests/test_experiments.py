"""Tests for the EDEL Experiments Engine."""

import json
import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from edel.experiments.registry import (
    register_experiment,
    get_experiment,
    list_experiments,
    load_from_file,
    _deep_merge,
)
from edel.experiments.runner import run_experiments, load_registry
from edel.experiments.analyzer import analyze_experiments, compare_experiments
from edel.experiments.metrics.segmentation import segmentation_metrics
from edel.experiments.metrics.embedding import embedding_metrics
from edel.experiments.metrics.operators import operator_metrics
from edel.experiments.metrics.structure import structure_metrics
from edel.config.defaults import RUN_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Generate a small synthetic dataset for metric testing."""
    N = 100
    rng = np.random.default_rng(42)
    dim = 8

    def fake_emb_col(offset=0):
        return [json.dumps((rng.standard_normal(dim) + offset).tolist()) for _ in range(N)]

    words = lambda n: ' '.join(['word'] * n)
    df = pd.DataFrame({
        'problem':        [words(rng.integers(10, 30)) for _ in range(N)],
        'method':         [words(rng.integers(10, 30)) for _ in range(N)],
        'finding':        [words(rng.integers(10, 30)) for _ in range(N)],
        'interpretation': [words(rng.integers(10, 30)) for _ in range(N)],
        'abstract_text':  [words(rng.integers(50, 100)) for _ in range(N)],
        'problem_embedding':        fake_emb_col(0),
        'method_embedding':         fake_emb_col(2),  # Shifted
        'finding_embedding':        fake_emb_col(5),  # Shifted more
        'interpretation_embedding': fake_emb_col(9),  # Shifted even more
    })
    return df, dim


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------

def test_registry_basic():
    """Test registration and retrieval of experiment configs."""
    name = "test_exp"
    config = {"key": "value"}
    register_experiment(name, config)
    
    assert name in list_experiments()
    assert get_experiment(name) == config
    
    # Test deep copy
    retrieved = get_experiment(name)
    retrieved["key"] = "changed"
    assert get_experiment(name)["key"] == "value"


def test_registry_error():
    """Test that missing experiments raise KeyError."""
    with pytest.raises(KeyError):
        get_experiment("non_existent")


def test_deep_merge():
    """Test recursive merging of configs."""
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    overrides = {"b": {"d": 4, "e": 5}, "f": 6}
    expected = {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}
    
    assert _deep_merge(base, overrides) == expected


def test_load_from_file(tmp_path):
    """Test bulk registration from a JSON file."""
    config_file = tmp_path / "experiments.json"
    data = {
        "exp1": {"embedding": {"model": "model1"}},
        "exp2": {"embedding": {"model": "model2"}}
    }
    config_file.write_text(json.dumps(data))
    
    names = load_from_file(config_file)
    assert "exp1" in names
    assert "exp2" in names
    assert get_experiment("exp1")["embedding"]["model"] == "model1"
    # Ensure defaults are merged
    assert "provider" in get_experiment("exp1")["data"]


# ---------------------------------------------------------------------------
# Runner Tests (Mocked)
# ---------------------------------------------------------------------------

@patch("edel.experiments.runner.run_full_pipeline")
def test_run_experiments(mock_pipeline, tmp_path):
    """Test the runner logic without executing the full pipeline."""
    configs = [
        {"data": {"provider": {"type": "openalex", "topic_id": "T1", "topic_name": "T1"}}},
        {"data": {"provider": {"type": "openalex", "topic_id": "T2", "topic_name": "T2"}}}
    ]
    
    records = run_experiments(configs, base_path=tmp_path)
    
    assert len(records) == 2
    assert mock_pipeline.call_count == 2
    assert records[0]["experiment_id"] == "openalex_T1_global"
    assert "data" in records[0]["artifact_refs"]
    
    # Check registry persistence
    registry_file = tmp_path / "experiments" / "registry.pkl"
    assert registry_file.exists()
    
    loaded = load_registry(tmp_path)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

def test_segmentation_metrics(synthetic_data):
    """Test text-level quality metrics."""
    df, _ = synthetic_data
    artifacts = {"structuring": df}
    
    result = segmentation_metrics(artifacts)
    metrics = result["metrics"]
    features = result["features"]
    
    assert "seg_ratio_mean" in metrics
    assert "abstract_len_mean" in metrics
    assert "dataset_size" in metrics
    assert metrics["dataset_size"] == len(df)
    assert "seg_ratio_dist" in features
    assert len(features["seg_ratio_dist"]) == len(df)


def test_embedding_metrics(synthetic_data):
    """Test cosine similarity and density metrics."""
    df, dim = synthetic_data
    artifacts = {"embedding": df, "_dimensions": dim}
    
    result = embedding_metrics(artifacts)
    metrics = result["metrics"]
    features = result["features"]
    
    assert "sim_pm" in metrics
    assert "density_p_mean" in metrics
    assert "sep_p_m" in metrics
    assert "sim_pm_dist" in features


def test_operator_and_structure_metrics(synthetic_data):
    """Test operator and structure metrics (shared context)."""
    df, dim = synthetic_data
    artifacts = {"embedding": df, "_dimensions": dim}
    
    # Must run operators first to populate context for structure
    op_result = operator_metrics(artifacts)
    assert "_operators" in artifacts
    assert "norm_pm" in op_result["metrics"]
    assert "transition_features" in op_result["features"]
    
    str_result = structure_metrics(artifacts)
    assert "silhouette_transitions" in str_result["metrics"]
    assert "silhouette_features" in str_result["metrics"]


# ---------------------------------------------------------------------------
# Analyzer & Comparison Tests
# ---------------------------------------------------------------------------

def test_analyze_experiments(synthetic_data, tmp_path):
    """Test the full analysis flow from records to DataFrame."""
    df, dim = synthetic_data
    
    # Mocking artifact loading to avoid filesystem hits
    with patch("edel.experiments.analyzer.load_artifact", return_value=df):
        experiments = [{
            "experiment_id": "test_exp",
            "config": RUN_CONFIG,
            "artifact_refs": {"embedding": MagicMock(), "structuring": MagicMock()}
        }]
        
        results_df = analyze_experiments(experiments, base_path=tmp_path)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 1
        assert "experiment_id" in results_df.columns
        assert "silhouette_transitions" in results_df.columns
        
        # Check feature persistence
        feature_file = tmp_path / "experiments" / "test_exp" / "features.pkl"
        assert feature_file.exists()


def test_compare_experiments(tmp_path):
    """Test pairwise KS tests between experiment features."""
    eid1 = "exp1"
    eid2 = "exp2"
    
    # Create fake feature artifacts
    f1 = {"cos_pm_mf_dist": np.random.rand(10)}
    f2 = {"cos_pm_mf_dist": np.random.rand(10)}
    
    for eid, feat in [(eid1, f1), (eid2, f2)]:
        feat_path = tmp_path / "experiments" / eid / "features.pkl"
        feat_path.parent.mkdir(parents=True)
        with open(feat_path, "wb") as f:
            pickle.dump(feat, f)
            
    experiments = [
        {"experiment_id": eid1},
        {"experiment_id": eid2}
    ]
    
    ks_df = compare_experiments(experiments, base_path=tmp_path, feature_dims=["cos_pm_mf_dist"])
    
    assert len(ks_df) == 1
    assert ks_df.iloc[0]["exp_a"] == eid1
    assert ks_df.iloc[0]["exp_b"] == eid2
    assert "ks_stat" in ks_df.columns
    assert "ks_pvalue" in ks_df.columns
