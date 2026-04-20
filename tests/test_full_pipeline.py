"""End-to-end test for the full EDEL pipeline."""

import shutil
import pytest
from edel.pipeline.run import run_full_pipeline
from edel.config.defaults import RUN_CONFIG


def test_full_pipeline_mock(tmp_path):
    """Run the entire 8-stage pipeline using mock providers and verify persistence."""
    config = RUN_CONFIG.copy()
    
    # Configure for a fast mock run
    config["data"]["provider"] = {
        "type": "scigen_null",
        "params": {"n_documents": 10}
    }
    config["structured_abstracts"]["provider"] = "mock"
    config["embedding"]["provider"] = "mock"
    config["embedding"]["n_dimensions"] = 16
    config["labeling"]["provider"] = "mock"
    
    # HDBSCAN needs enough points for min_cluster_size
    config["clustering"] = {
        "test_cluster": {
            "source": "proj_p",
            "algorithm": "kmeans",
            "params": {"n_clusters": 2, "n_init": 10}
        }
    }
    config["labeling"]["clusters"]["cluster_keys"] = ["test_cluster"]

    # 1. First run: should compute everything
    print("\n--- FIRST RUN ---")
    results1 = run_full_pipeline(config, base_path=tmp_path)
    
    assert "data" in results1
    assert "structuring" in results1
    assert "embedding" in results1
    assert "projection" in results1
    assert "vector_field" in results1
    assert "clustering_df" in results1
    assert "labels" in results1
    assert "landscape" in results1
    
    # Verify some output properties
    df = results1["clustering_df"]
    assert "cluster_test_cluster" in df.columns
    assert "proposed_label" in results1["labels"]["clusters"]["test_cluster"][0]
    assert "z" in results1["landscape"]["terrain"]

    # 2. Second run: should load everything from artifacts
    print("\n--- SECOND RUN (Should use cache) ---")
    results2 = run_full_pipeline(config, base_path=tmp_path)
    
    # Verify results are the same
    assert results2["labels"] == results1["labels"]
    
    # 3. Third run with force: should recompute
    print("\n--- THIRD RUN (Force recompute) ---")
    results3 = run_full_pipeline(config, base_path=tmp_path, force=True)
    assert results3["labels"] == results1["labels"] # Mocks return same data
