import sys
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from edel.il.index import NumpyRAGIndex

# Add scripts to path so we can import build_afp_index
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
import build_afp_index  # type: ignore


def test_build_afp_index_calculate_missing_embeddings(tmp_path, monkeypatch):
    # 1. Create a dummy index with metadata but NO embeddings (skipped state)
    output_dir = tmp_path / "rag_index"
    output_dir.mkdir()
    
    metadata = [
        {
            "title": "HOL.List.append_Nil",
            "problem": "[] @ ys = ys",
            "method": "by simp",
            "finding": "by simp",
            "interpretation": "[] @ ys = ys",
            "theory": "HOL.List",
            "keyword": "lemma",
        },
        {
            "title": "HOL.List.append_assoc",
            "problem": "(xs @ ys) @ zs = xs @ (ys @ zs)",
            "method": "by simp",
            "finding": "by simp",
            "interpretation": "(xs @ ys) @ zs = xs @ (ys @ zs)",
            "theory": "HOL.List",
            "keyword": "lemma",
        }
    ]
    
    # Save the index with metadata and empty/None embeddings
    meta_df = pd.DataFrame(metadata)
    meta_df.to_parquet(output_dir / "metadata.parquet", index=False)
    
    # Save empty npz for embeddings
    npz_kwargs = {}
    np.savez_compressed(output_dir / "embeddings.npz", **npz_kwargs)
    
    # 2. Mock run_embedding_stage
    def mock_run_embedding_stage(df, config, *args, **kwargs):
        # Return df with fake embeddings for each aspect (using 128 dimension for testing)
        df_out = df.copy()
        for aspect in ["problem", "method", "finding", "interpretation"]:
            # Generate dummy embedding arrays for the number of rows in df
            df_out[f"{aspect}_embedding"] = [json.dumps([1.0] * 128) for _ in range(len(df))]
        return df_out
        
    monkeypatch.setattr(build_afp_index, "run_embedding_stage", mock_run_embedding_stage)
    
    # Mock compute_and_save_landscape_height
    import edel.il.compute_landscape_height
    mock_compute = MagicMock()
    monkeypatch.setattr(edel.il.compute_landscape_height, "compute_and_save_landscape_height", mock_compute)
    
    # Set mock environment variables
    monkeypatch.setenv("VOYAGE_API_KEY", "dummy_api_key")
    
    # 3. Call build_afp_index.main() with --calculate-missing-embeddings
    test_args = [
        "build_afp_index.py",
        "--output", str(output_dir),
        "--calculate-missing-embeddings",
        "--provider", "voyage",
        "--model", "voyage-code-3",
    ]
    
    with patch("sys.argv", test_args):
        build_afp_index.main()
        
    # 4. Verify index was updated and saved
    master_index = NumpyRAGIndex()
    master_index.load(output_dir)
    
    assert len(master_index.metadata) == 2
    for aspect in ["problem", "method", "finding", "interpretation"]:
        assert master_index.embeddings[aspect] is not None
        assert master_index.embeddings[aspect].shape == (2, 128)
        assert np.allclose(master_index.embeddings[aspect], 1.0)
        
    # Verify landscape height was computed
    mock_compute.assert_called_once_with(output_dir)


def test_build_afp_index_calculate_missing_embeddings_resume(tmp_path, monkeypatch):
    # Test that we can resume calculation if some are already embedded
    output_dir = tmp_path / "rag_index"
    output_dir.mkdir()
    
    metadata = [
        {
            "title": "HOL.List.append_Nil",
            "problem": "[] @ ys = ys",
            "method": "by simp",
            "finding": "by simp",
            "interpretation": "[] @ ys = ys",
            "theory": "HOL.List",
            "keyword": "lemma",
        },
        {
            "title": "HOL.List.append_assoc",
            "problem": "(xs @ ys) @ zs = xs @ (ys @ zs)",
            "method": "by simp",
            "finding": "by simp",
            "interpretation": "(xs @ ys) @ zs = xs @ (ys @ zs)",
            "theory": "HOL.List",
            "keyword": "lemma",
        }
    ]
    
    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_parquet(output_dir / "metadata.parquet", index=False)
    
    # Save embeddings where the first item already has an embedding, but the second is missing
    # (i.e. length of embeddings is 1, length of metadata is 2)
    npz_kwargs = {}
    for aspect in ["problem", "method", "finding", "interpretation"]:
        npz_kwargs[aspect] = np.array([[2.0] * 128], dtype=np.float32)
    np.savez_compressed(output_dir / "embeddings.npz", **npz_kwargs)
    
    # Mock run_embedding_stage
    def mock_run_embedding_stage(df, config, *args, **kwargs):
        # We only expect the second item (index 1) to be embedded
        assert len(df) == 1
        assert df.iloc[0]["title"] == "HOL.List.append_assoc"
        df_out = df.copy()
        for aspect in ["problem", "method", "finding", "interpretation"]:
            df_out[f"{aspect}_embedding"] = [json.dumps([3.0] * 128)]
        return df_out
        
    monkeypatch.setattr(build_afp_index, "run_embedding_stage", mock_run_embedding_stage)
    
    # Mock compute_and_save_landscape_height
    import edel.il.compute_landscape_height
    mock_compute = MagicMock()
    monkeypatch.setattr(edel.il.compute_landscape_height, "compute_and_save_landscape_height", mock_compute)
    
    # Set mock environment variables
    monkeypatch.setenv("VOYAGE_API_KEY", "dummy_api_key")
    
    test_args = [
        "build_afp_index.py",
        "--output", str(output_dir),
        "--calculate-missing-embeddings",
        "--provider", "voyage",
        "--model", "voyage-code-3",
    ]
    
    with patch("sys.argv", test_args):
        build_afp_index.main()
        
    # Verify index was updated and saved
    master_index = NumpyRAGIndex()
    master_index.load(output_dir)
    
    assert len(master_index.metadata) == 2
    for aspect in ["problem", "method", "finding", "interpretation"]:
        assert master_index.embeddings[aspect] is not None
        assert master_index.embeddings[aspect].shape == (2, 128)
        # First row should be 2.0 (untouched), second row should be 3.0 (newly computed)
        assert np.allclose(master_index.embeddings[aspect][0], 2.0)
        assert np.allclose(master_index.embeddings[aspect][1], 3.0)


def test_build_afp_index_token_failure(tmp_path, monkeypatch):
    # Mock sessions to run
    monkeypatch.setattr(build_afp_index, "get_afp_sessions", lambda x: ["TestSession"])
    monkeypatch.setattr(build_afp_index, "build_session_heap", lambda *a: True)
    
    # Mock subprocess.Popen to return a mock process whose stdout is empty (no token retrieved)
    mock_proc = MagicMock()
    mock_proc.stdout.readline.return_value = b""
    monkeypatch.setattr(build_afp_index.subprocess, "Popen", lambda *a, **kw: mock_proc)
    
    # Set args
    test_args = [
        "build_afp_index.py",
        "--output", str(tmp_path),
        "--skip-embedding",
    ]
    
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            build_afp_index.main()
        assert exc_info.value.code == 1


def test_build_afp_index_ingestion_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(build_afp_index, "get_afp_sessions", lambda x: ["TestSession"])
    monkeypatch.setattr(build_afp_index, "build_session_heap", lambda *a: True)
    
    # Mock Popen to return a valid token line
    mock_proc = MagicMock()
    # First readline returns the token log, second returns empty
    mock_proc.stdout.readline.side_effect = [
        b"IR_Repl.token: secret_token_123\n",
        "● REPL ready. Waiting for connections on 127.0.0.1:9147\n".encode("utf-8"),
        b""
    ]
    monkeypatch.setattr(build_afp_index.subprocess, "Popen", lambda *a, **kw: mock_proc)
    
    # Mock ingest_session_lemmas to raise an exception
    def mock_ingest(*args, **kwargs):
        raise ConnectionRefusedError("Could not connect to daemon")
        
    monkeypatch.setattr(build_afp_index, "ingest_session_lemmas", mock_ingest)
    
    # Set args
    test_args = [
        "build_afp_index.py",
        "--output", str(tmp_path),
        "--skip-embedding",
    ]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            build_afp_index.main()
        assert exc_info.value.code == 1


def test_build_afp_index_include_hol_and_deduplicate(tmp_path, monkeypatch):
    # 1. Setup existing index with one existing lemma ("HOL.List.existing_lemma")
    output_dir = tmp_path / "rag_index"
    output_dir.mkdir()
    
    master_index = NumpyRAGIndex()
    master_index.metadata = [{"title": "HOL.List.existing_lemma"}]
    for aspect in ["problem", "method", "finding", "interpretation"]:
        master_index.embeddings[aspect] = np.zeros((1, 128), dtype=np.float32)
    master_index.save(output_dir)
    
    # 2. Mock session/heap helper functions
    monkeypatch.setattr(build_afp_index, "get_afp_sessions", lambda x: ["TestSession"])
    monkeypatch.setattr(build_afp_index, "build_session_heap", lambda *a: True)
    
    # Mock Popen to return valid token
    mock_proc = MagicMock()
    mock_proc.stdout.readline.side_effect = [
        b"IR_Repl.token: secret_token_123\n",
        "● REPL ready. Waiting for connections on 127.0.0.1:9147\n".encode("utf-8"),
        b""
    ]
    monkeypatch.setattr(build_afp_index.subprocess, "Popen", lambda *a, **kw: mock_proc)
    
    # Track the theory_filter that was passed to ingest_session_lemmas
    passed_filter = None
    
    def mock_ingest(host, port, token, theory_filter):
        nonlocal passed_filter
        passed_filter = theory_filter
        
        # Return two lemmas: one already exists ("HOL.List.existing_lemma") and one is new ("TestSession.new_lemma")
        return pd.DataFrame([
            {
                "title": "HOL.List.existing_lemma",
                "problem": "existing problem",
                "method": "existing method",
                "finding": "existing finding",
                "interpretation": "existing interpretation",
                "theory": "HOL.List",
                "keyword": "lemma",
            },
            {
                "title": "TestSession.new_lemma",
                "problem": "new problem",
                "method": "new method",
                "finding": "new finding",
                "interpretation": "new interpretation",
                "theory": "TestSession.Theory",
                "keyword": "lemma",
            }
        ])
        
    monkeypatch.setattr(build_afp_index, "ingest_session_lemmas", mock_ingest)
    
    # Mock landscape height computation
    import edel.il.compute_landscape_height
    mock_compute = MagicMock()
    monkeypatch.setattr(edel.il.compute_landscape_height, "compute_and_save_landscape_height", mock_compute)
    
    # Set arguments with --include-hol and --skip-embedding
    test_args = [
        "build_afp_index.py",
        "--output", str(output_dir),
        "--skip-embedding",
        "--include-hol",
    ]
    
    with patch("sys.argv", test_args):
        build_afp_index.main()
        
    # Verify that the regex included both TestSession and HOL
    assert passed_filter == "^(?:TestSession|HOL|HOL-[a-zA-Z0-9_-]+)\\."
    
    # Load index to verify deduplication worked: the existing lemma should NOT be duplicated, 
    # and the new lemma should be appended. Total metadata items should be 2.
    updated_index = NumpyRAGIndex()
    updated_index.load(output_dir)
    
    assert len(updated_index.metadata) == 2
    assert updated_index.metadata[0]["title"] == "HOL.List.existing_lemma"
    assert updated_index.metadata[1]["title"] == "TestSession.new_lemma"

