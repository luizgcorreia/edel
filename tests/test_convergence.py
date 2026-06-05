"""Unit tests for convergence analysis and calibration."""

from __future__ import annotations

import json
from pathlib import Path
import unittest.mock
import numpy as np
import pandas as pd
import pytest

from edel.analysis.convergence import sample_temporal_stratified, run_convergence_analysis


def test_sample_temporal_stratified():
    """Verify that sample_temporal_stratified preserves temporal proportions."""
    # Create synthetic dataframe with a specific year distribution
    # 10 papers in 2020, 20 papers in 2021 (1:2 ratio)
    data = {
        "publication_year": [2020] * 10 + [2021] * 20
    }
    df = pd.DataFrame(data)
    
    # Sample n = 15 papers (should result in 5 from 2020 and 10 from 2021)
    sampled_indices = sample_temporal_stratified(df, 15, random_state=42)
    assert len(sampled_indices) == 15
    
    sampled_years = df.iloc[sampled_indices]["publication_year"].values.tolist()
    count_2020 = sampled_years.count(2020)
    count_2021 = sampled_years.count(2021)
    
    assert count_2020 == 5
    assert count_2021 == 10


@pytest.fixture
def mock_experiment_setup(tmp_path) -> tuple[str, Path]:
    """Set up temporary configs and parquet files for testing the convergence runner."""
    experiment_id = "test_experiment"
    
    # Define config mirroring standard runs
    config = {
        "random_seed": 42,
        "embedding_mode": "aspects",
        "data": {
            "provider": {
                "type": "test_provider",
                "topic_id": "test_topic",
                "region": "global"
            }
        },
        "embedding": {
            "n_dimensions": 8
        }
    }
    
    # Parquet name based on hash and label
    # label = get_experiment_label(config) = "test_provider_test_topic_global"
    label = "test_provider_test_topic_global"
    
    # Save a small synthetic embeddings dataframe
    np.random.seed(42)
    N = 600
    dims = 8
    
    prob = np.random.randn(N, dims)
    meth = prob + np.random.randn(N, dims) * 0.1
    find = meth + np.random.randn(N, dims) * 0.1
    interp = find + np.random.randn(N, dims) * 0.1
    
    data = {
        "publication_year": [2020 + (i % 5) for i in range(N)],
        "problem_embedding": [json.dumps(p.tolist()) for p in prob],
        "method_embedding": [json.dumps(m.tolist()) for m in meth],
        "finding_embedding": [json.dumps(f.tolist()) for f in find],
        "interpretation_embedding": [json.dumps(i.tolist()) for i in interp],
    }
    df = pd.DataFrame(data)
    
    # We need to compute stage_hash to write to the correct file name.
    # To bypass deterministic hash resolving in make_stage_artifact,
    # we can mock or just write to the path it resolves to.
    # Let's check where it writes to by running a mock stage artifact call.
    return experiment_id, config, df


def test_run_convergence_analysis(tmp_path, mock_experiment_setup):
    """Test convergence analysis executing properly on mock data."""
    experiment_id, config, df = mock_experiment_setup
    
    # Create the target directory structure
    label = "test_provider_test_topic_global"
    emb_dir = tmp_path / "embeddings" / label
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parquet to embeddings dir
    # We will mock make_stage_artifact to return a mock artifact pointing to our file
    with unittest.mock.patch("edel.analysis.convergence.get_experiment", return_value=config), \
         unittest.mock.patch("edel.analysis.convergence.make_stage_artifact") as mock_artifact_fn:
         
        # Set up mock artifact
        mock_art = unittest.mock.MagicMock()
        mock_art.parquet_path = emb_dir / "embeddings_test.parquet"
        mock_artifact_fn.return_value = mock_art
        
        # Save parquet
        df.to_parquet(mock_art.parquet_path, index=False)
        
        # Run convergence analysis
        results = run_convergence_analysis(experiment_id, base_path=tmp_path, force=True)
        
        # Basic validation
        assert results["experiment_id"] == experiment_id
        assert results["N"] == 600
        
        # Verify H1 results structure
        h1_res = results["h1_results"]
        assert "sample_sizes" in h1_res
        assert len(h1_res["sample_sizes"]) > 0
        # KS statistic and Wasserstein values present
        for key in ["norm_pm", "norm_mf", "norm_fi", "cos_pm_mf", "cos_pm_fi", "cos_mf_fi"]:
            assert key in h1_res["full_refs"]
            assert key in h1_res["data"][h1_res["sample_sizes"][0]]["ks_stat"]
            assert key in h1_res["data"][h1_res["sample_sizes"][0]]["w_dist"]
            
        # Verify H2 results structure
        h2_res = results["h2_results"]
        assert "mae_z" in h2_res["data"][h2_res["sample_sizes"][0]]
        assert "jaccard" in h2_res["data"][h2_res["sample_sizes"][0]]
        
        # Verify H3 results structure
        h3_res = results["h3_results"]
        assert len(h3_res["percentages"]) == 5
        assert "Scheme A" in h3_res["data"]
        assert "Scheme B" in h3_res["data"]
