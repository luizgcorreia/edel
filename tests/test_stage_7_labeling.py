"""Tests for Stage 7: Labeling."""

import os
import pandas as pd
import pytest
from edel.pipeline.labeling import run_labeling_stage
from edel.io.llm import MockClient, OpenAIClient, LMStudioClient


@pytest.fixture
def df_final():
    # 10 documents to allow sampling
    texts = [
        "Research on neural network architectures for vision.",
        "Deep learning applications in medicine.",
        "Convolutional networks for image recognition.",
        "Transformers in natural language processing.",
        "Reinforcement learning for robotics.",
        "Quantum computing algorithms for optimization.",
        "Superconducting qubits and error correction.",
        "Quantum entanglement in many-body systems.",
        "Topological insulators in condensed matter.",
        "Photonics and quantum communication.",
    ]
    return pd.DataFrame(
        {
            "abstract_text": texts,
            "proj_problem_umap_x": [0.1, 0.15, 0.12, 0.18, 0.2, 0.8, 0.85, 0.82, 0.9, 0.95],
            "proj_problem_umap_y": [0.1, 0.15, 0.12, 0.18, 0.2, 0.8, 0.85, 0.82, 0.9, 0.95],
            "cluster_domains": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def test_labeling_mock(df_final):
    field = pd.DataFrame()
    config = {
        "labeling": {
            "axis": {"enabled": True, "projection": "umap", "n_samples": 2},
            "clusters": {"enabled": True, "cluster_keys": ["domains"], "n_samples": 2},
        }
    }
    # Mock response must be valid JSON for both axis and cluster tasks
    mock_json = '{"cluster_topics": "AI", "proposed_label": "Artificial Intelligence", "axis_label": "Scale", "negative_pole": "Local", "positive_pole": "Global"}'
    client = MockClient(response=mock_json)

    results = run_labeling_stage(df_final, field, config, client)
    assert "clusters" in results
    assert "domains" in results["clusters"]
    assert 0 in results["clusters"]["domains"]
    assert results["clusters"]["domains"][0]["proposed_label"] == "Artificial Intelligence"
    assert len(results["axes"]) == 2  # Axis 0 and 1


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_labeling_openai_integration(df_final):
    """Integration test with OpenAI."""
    field = pd.DataFrame()
    config = {
        "labeling": {
            "model": "gpt-5-mini",
            "axis": {"enabled": True, "n_samples": 2},
            "clusters": {"enabled": True, "cluster_keys": ["domains"], "n_samples": 2},
        }
    }
    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    results = run_labeling_stage(df_final, field, config, client)

    assert "clusters" in results
    assert "domains" in results["clusters"]
    # Check if we got real labels back
    assert "proposed_label" in results["clusters"]["domains"][0]


def test_labeling_lmstudio_integration(df_final):
    """Integration test with LM Studio."""
    import httpx

    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    try:
        httpx.get(base_url.replace("/v1", "/"))
    except Exception:
        pytest.skip("LM Studio server not found at " + base_url)

    field = pd.DataFrame()
    config = {
        "labeling": {
            "model": "gemma-4-e4b-it",
            "axis": {"enabled": True, "n_samples": 2},
            "clusters": {"enabled": True, "cluster_keys": ["domains"], "n_samples": 2},
        }
    }
    client = LMStudioClient(base_url=base_url)
    results = run_labeling_stage(df_final, field, config, client)

    assert "clusters" in results
    assert "domains" in results["clusters"]
    assert "proposed_label" in results["clusters"]["domains"][0]
