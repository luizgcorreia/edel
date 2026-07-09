"""Unit tests for the build_index script."""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from edel.il import build_il_index

def test_build_index_main(tmp_path, monkeypatch):
    # Mocking ingest_session_lemmas
    mock_df = pd.DataFrame([
        {
            "title": "HOL.List.append_Nil",
            "problem": "[] @ ys = ys",
            "method": "Theory List",
            "finding": "simp",
            "interpretation": "none",
            "theory": "HOL.List",
            "file": "List.thy",
            "line": 10,
        }
    ])
    
    # Mock run_embedding_stage to return df with embedding columns
    mock_df_embedded = mock_df.copy()
    mock_df_embedded["problem_embedding"] = json.dumps([1.0, 0.0])
    mock_df_embedded["method_embedding"] = json.dumps([0.0, 1.0])
    mock_df_embedded["finding_embedding"] = json.dumps([0.5, 0.5])
    mock_df_embedded["interpretation_embedding"] = json.dumps([0.1, 0.9])
    
    # Mocking functions
    monkeypatch.setattr(build_il_index, "ingest_session_lemmas", lambda **kwargs: mock_df)
    monkeypatch.setattr(build_il_index, "run_embedding_stage", lambda df, config: mock_df_embedded)
    
    output_dir = tmp_path / "test_rag_index"
    
    # Run build_index with mocked args
    test_args = [
        "build_il_index.py",
        "--token", "dummy-token",
        "--output", str(output_dir),
        "--provider", "openai",
        "--model", "test-model"
    ]
    
    with patch("sys.argv", test_args):
        build_il_index.main()
        
    # Check that output files were created
    assert (output_dir / "metadata.parquet").exists()
    assert (output_dir / "embeddings.npz").exists()
