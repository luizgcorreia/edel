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
    """Test Stage 3 filters out rows with empty fields."""
    df = pd.DataFrame([
        {"problem": "P1", "method": "", "finding": "F1", "interpretation": " "},
    ])
    df_embedded = run_embedding_stage(df, base_run_config)
    
    assert len(df_embedded) == 0


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


def test_embedding_stage_filtering():
    """Test that empty or missing aspects are filtered and the report is generated correctly."""
    # Create test dataframe with a mix of valid and invalid/missing entries
    df = pd.DataFrame([
        # 1. Completely valid
        {"title": "Valid", "problem": "P1", "method": "M1", "finding": "F1", "interpretation": "I1"},
        # 2. Missing method (empty string)
        {"title": "Missing method", "problem": "P2", "method": "", "finding": "F2", "interpretation": "I2"},
        # 3. Missing problem (NaN)
        {"title": "Missing problem", "problem": None, "method": "M3", "finding": "F3", "interpretation": "I3"},
        # 4. Missing interpretation (whitespace only)
        {"title": "Missing interpretation", "problem": "P4", "method": "M4", "finding": "F4", "interpretation": "   "},
    ])
    
    config = {
        "processing_mode": "simple",
        "embedding": {
            "provider": "mock",
            "model": "text-embedding-ada-002",
            "mode": "multi",
        },
    }
    
    # Run with return_report=True
    df_embedded, report = run_embedding_stage(df, config, return_report=True)
    
    # Only the first entry should stay
    assert len(df_embedded) == 1
    assert df_embedded["title"].iloc[0] == "Valid"
    
    # Check report content
    assert report["initial_count"] == 4
    assert report["final_count"] == 1
    assert report["total_filtered"] == 3
    
    # Aspect specific coverage
    cov = report["aspect_coverage"]
    assert cov["problem"]["filtered"] == 1
    assert cov["problem"]["stayed"] == 3
    assert cov["method"]["filtered"] == 1
    assert cov["method"]["stayed"] == 3
    assert cov["finding"]["filtered"] == 0
    assert cov["finding"]["stayed"] == 4
    assert cov["interpretation"]["filtered"] == 1
    assert cov["interpretation"]["stayed"] == 3


def test_voyage_client_init_and_generate():
    """Test VoyageClient initialization."""
    from edel.io.llm import get_llm_client, VoyageClient
    config = {
        "provider": "voyage",
        "model": "voyage-code-2",
        "api_key": "test-key",
        "input_type": "document"
    }
    client = get_llm_client(config)
    assert isinstance(client, VoyageClient)
    assert client.model == "voyage-code-2"
    assert client.api_key == "test-key"


def test_voyage_client_embedding(monkeypatch):
    """Test VoyageClient embedding generation using mocked requests."""
    from edel.io.llm import VoyageClient
    import requests
    
    class MockResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            
    def mock_post(url, headers, json, timeout):
        assert url == "https://api.voyageai.com/v1/embeddings"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "voyage-code-2"
        assert json["input"] == ["hello"]
        assert json["input_type"] == "document"
        return MockResponse()
        
    monkeypatch.setattr(requests, "post", mock_post)
    
    client = VoyageClient(model="voyage-code-2", api_key="test-key", input_type="document")
    emb = client.generate_embedding("hello")
    assert emb == [0.1, 0.2, 0.3]


def test_embedding_deduplication_optimization(df_structured, base_run_config):
    """Test that embedding stage correctly deduplicates identical aspect values to minimize API calls."""
    from unittest.mock import MagicMock
    from edel.io.llm import MockClient
    from edel.pipeline.embedding import run_embedding_stage

    # Create a DataFrame with duplicate aspect values
    # e.g. "by simp" is shared, and unconditional lemma has P=I
    df = pd.DataFrame([
        # Lemma 1: Tactic proof where method = finding = "by simp"
        {
            "problem": "A ⟹ B",
            "method": "by simp",
            "finding": "by simp",
            "interpretation": "B",
        },
        # Lemma 2: Unconditional lemma where problem = interpretation = "x = y"
        {
            "problem": "x = y",
            "method": "by simp",
            "finding": "by simp",
            "interpretation": "x = y",
        },
    ])

    # Total aspect slots: 2 rows * 4 aspects = 8 slots
    # Unique text aspects: {"A ⟹ B", "by simp", "B", "x = y"} -> exactly 4 unique strings.

    # We spy on generate_embedding or create_batch by mocking the client
    spy_client = MockClient()
    original_generate = spy_client.generate_embedding
    called_texts = []

    def mock_generate(text, **kwargs):
        if isinstance(text, list):
            called_texts.extend(text)
        else:
            called_texts.append(text)
        return original_generate(text, **kwargs)

    spy_client.generate_embedding = mock_generate

    # Mock get_llm_client to return our spy_client
    import edel.pipeline.embedding
    from unittest.mock import patch
    with patch("edel.pipeline.embedding.get_llm_client", return_value=spy_client):
        df_embedded = run_embedding_stage(df, base_run_config)

    # Verify output embeddings are correctly mapped to all columns
    assert "problem_embedding" in df_embedded.columns
    assert "method_embedding" in df_embedded.columns
    assert "finding_embedding" in df_embedded.columns
    assert "interpretation_embedding" in df_embedded.columns

    # Verify that the generated embeddings for identical text are exactly equal
    # Row 0: method = finding = "by simp"
    assert df_embedded["method_embedding"].iloc[0] == df_embedded["finding_embedding"].iloc[0]
    # Row 1: problem = interpretation = "x = y", method = finding = "by simp"
    assert df_embedded["problem_embedding"].iloc[1] == df_embedded["interpretation_embedding"].iloc[1]
    assert df_embedded["method_embedding"].iloc[1] == df_embedded["finding_embedding"].iloc[1]
    assert df_embedded["method_embedding"].iloc[0] == df_embedded["method_embedding"].iloc[1]

    # Verify that only the 4 unique texts were actually sent to the embedding model
    unique_called = set(called_texts)
    assert len(unique_called) == 4
    assert unique_called == {
        "Theory: Unknown | Lemma: unnamed | Premises:\nA ⟹ B",
        "Theory: Unknown | Lemma: unnamed | Proof:\nby simp",
        "Theory: Unknown | Lemma: unnamed | Conclusion:\nB",
        "Theory: Unknown | Lemma: unnamed | Statement:\nx = y",
    }
    assert len(called_texts) == 4, f"Expected 4 total embedding generations (deduplicated), but got {len(called_texts)}: {called_texts}"



