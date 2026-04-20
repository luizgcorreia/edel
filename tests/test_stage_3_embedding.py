"""Tests for Stage 3: Text Embedding."""

import json
import os
import pandas as pd
import pytest
from edel.pipeline.embedding import run_embedding_stage


@pytest.fixture
def base_run_config():
    return {
        "processing_mode": "simple",
        "embedding": {
            "provider": "mock",
            "model": "text-embedding-ada-002",
            "mode": "multi",
            "batch_size": 2,
        },
    }


@pytest.fixture
def df_structured():
    return pd.DataFrame(
        [
            {
                "title": "Title 1",
                "problem": "Problem 1",
                "method": "Method 1",
                "finding": "Finding 1",
                "interpretation": "Interpretation 1",
            },
            {
                "title": "Title 2",
                "problem": "Problem 2",
                "method": "Method 2",
                "finding": "Finding 2",
                "interpretation": "Interpretation 2",
            },
        ]
    )


def test_embedding_stage_simple_multi(df_structured, base_run_config):
    """Test Stage 3 in simple multi mode."""
    df_embedded = run_embedding_stage(df_structured, base_run_config)

    assert isinstance(df_embedded, pd.DataFrame)
    assert "problem_embedding" in df_embedded.columns
    assert "method_embedding" in df_embedded.columns
    
    # Check that it's a JSON string of a list
    emb_str = df_embedded["problem_embedding"].iloc[0]
    assert isinstance(emb_str, str)
    emb = json.loads(emb_str)
    assert isinstance(emb, list)
    assert len(emb) == 1536


def test_embedding_stage_simple_single(df_structured, base_run_config):
    """Test Stage 3 in simple single mode."""
    base_run_config["embedding"]["mode"] = "single"
    df_embedded = run_embedding_stage(df_structured, base_run_config)

    assert isinstance(df_embedded, pd.DataFrame)
    assert "embedding" in df_embedded.columns
    assert "problem_embedding" not in df_embedded.columns


def test_embedding_stage_batch_multi(df_structured, base_run_config):
    """Test Stage 3 in batch multi mode."""
    base_run_config["processing_mode"] = "batch"
    df_embedded = run_embedding_stage(df_structured, base_run_config)

    assert isinstance(df_embedded, pd.DataFrame)
    assert "problem_embedding" in df_embedded.columns
    assert len(df_embedded) == 2
    
    emb_str = df_embedded["problem_embedding"].iloc[0]
    assert len(json.loads(emb_str)) == 1536


def test_embedding_stage_empty_texts(base_run_config):
    """Test Stage 3 with some empty fields."""
    df = pd.DataFrame([
        {"problem": "P1", "method": "", "finding": "F1", "interpretation": " "},
    ])
    df_embedded = run_embedding_stage(df, base_run_config)
    
    assert df_embedded["problem_embedding"].iloc[0] is not None
    # Empty fields should be None (as JSON string "null" or Python None depending on impl)
    # My impl: col_data.append(json.dumps(res) if res is not None else None)
    # So it should be Python None
    assert df_embedded["method_embedding"].iloc[0] is None
    assert df_embedded["interpretation_embedding"].iloc[0] is None


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_embedding_openai_integration():
    """Integration test with real OpenAI."""
    df = pd.DataFrame([{"problem": "P1", "method": "M1", "finding": "F1", "interpretation": "I1"}])
    config = {
        "processing_mode": "simple",
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-ada-002",
            "mode": "multi",
        },
    }
    df_embedded = run_embedding_stage(df, config)
    assert len(df_embedded) == 1
    emb = json.loads(df_embedded["problem_embedding"].iloc[0])
    assert len(emb) == 1536


def test_embedding_lmstudio_integration():
    """Integration test with LM Studio."""
    import httpx
    import os

    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    try:
        httpx.get(base_url.replace("/v1", "/"))
    except Exception:
        pytest.skip("LM Studio server not found at " + base_url)

    df = pd.DataFrame([{"problem": "P1", "method": "M1", "finding": "F1", "interpretation": "I1"}])
    config = {
        "processing_mode": "simple",
        "embedding": {
            "provider": "lmstudio",
            "model": "text-embedding-qwen3-embedding-0.6b",
            "mode": "multi",
        },
    }
    df_embedded = run_embedding_stage(df, config)
    assert len(df_embedded) == 1
    emb = json.loads(df_embedded["problem_embedding"].iloc[0])
    # Verify dimensions (user mentioned 1024 for Qwen3)
    assert len(emb) == 1024
