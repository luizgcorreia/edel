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


def test_full_pipeline_afp_rag(tmp_path):
    """Run the pipeline end-to-end with the afp_rag provider and 'none' for structuring and embedding."""
    import numpy as np
    from edel.il.index import NumpyRAGIndex
    
    # 1. Create a dummy index with 5 lemmas
    index = NumpyRAGIndex()
    index.metadata = [
        {
            "title": f"Session.Theory.lemma_{i}",
            "problem": f"lemma_{i} statement",
            "method": f"lemma_{i} context",
            "finding": f"lemma_{i} strategy",
            "interpretation": "none",
            "theory": "Session.Theory",
            "file": "Theory.thy",
            "line": i * 10,
            "proof_text": "by simp",
            "statement_text": f"lemma lemma_{i}"
        }
        for i in range(5)
    ]
    # Set dummy embeddings (dim=16)
    for aspect in ["problem", "method", "finding", "interpretation"]:
        index.embeddings[aspect] = np.random.rand(5, 16).astype(np.float32)
        
    index_dir = tmp_path / "dummy_index"
    index.save(index_dir)
    
    # 2. Configure pipeline
    config = RUN_CONFIG.copy()
    config["data"]["provider"] = {
        "type": "afp_rag",
        "params": {"index_dir": str(index_dir)}
    }
    # Set structuring and embedding to 'none'
    config["structured_abstracts"] = {
        "provider": "none"
    }
    config["embedding"] = {
        "mode": "multi",
        "provider": "none",
        "n_dimensions": 16
    }
    config["labeling"]["provider"] = "mock"
    config["clustering"] = {
        "test_cluster": {
            "source": "proj_p",
            "algorithm": "kmeans",
            "params": {"n_clusters": 2, "n_init": 10}
        }
    }
    config["labeling"]["clusters"]["cluster_keys"] = ["test_cluster"]
    
    # Run full pipeline
    results = run_full_pipeline(config, base_path=tmp_path / "pipeline_run")
    
    assert "data" in results
    assert "structuring" in results
    assert "embedding" in results
    assert "projection" in results
    assert "vector_field" in results
    assert "clustering_df" in results
    assert "labels" in results
    assert "landscape" in results
