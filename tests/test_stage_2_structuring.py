"""Tests for Stage 2: Structured Abstracts."""

import os
import pandas as pd
import pytest
from edel.io.artifact import load_artifact, make_stage_artifact, save_artifact
from edel.pipeline.data import run_data_stage
from edel.pipeline.structuring import run_structuring_stage


@pytest.fixture
def run_config():
    """Test configuration for Stage 1 and 2."""
    return {
        "data": {
            "provider": {
                "type": "lexicon_null",
                "topic_id": "TTEST",
                "topic_name": "Test Topic",
                "params": {"n_documents": 5},
            }
        },
        "structured_abstracts": {
            "provider": "mock",
            "model": "mock-model",
            "min_sentences": 1,
            "min_tokens": 5,
        },
        "processing_mode": "simple",
    }


def ensure_stage_1_artifacts(config, base_path):
    """Run Stage 1 only if artifacts are missing."""
    artifact = make_stage_artifact(config, base_path, "data_collection", "dataset")
    if not artifact.parquet_path.exists():
        df, _ = run_data_stage(config["data"])
        save_artifact(artifact, df)
    return load_artifact(artifact)


def test_structuring_stage_mock(run_config, tmp_path):
    """Test Stage 2 using the mock LLM client."""
    # 1. Setup Stage 1 dependency
    df_stage1 = ensure_stage_1_artifacts(run_config, tmp_path)
    assert len(df_stage1) == 5

    # 2. Run Stage 2
    df_stage2, report = run_structuring_stage(df_stage1, run_config)

    # 3. Verify results
    assert isinstance(df_stage2, pd.DataFrame)
    assert len(df_stage2) == 5
    for col in ["problem", "method", "finding", "interpretation"]:
        assert col in df_stage2.columns
        # Mock client should have prefilled these
        assert "Mock" in df_stage2[col].iloc[0]


def test_structuring_merging_logic():
    """Test that LLM snippets are correctly merged with existing provider data."""
    from edel.pipeline.structuring import parse_and_merge_results

    df = pd.DataFrame(
        [
            {
                "title": "Test Title",
                "abstract": "Test Abstract",
                "problem": "Existing Problem",
                "method": "Existing Method",
            }
        ]
    )
    df.index = [10]  # Non-zero index to test ID mapping

    results = {
        "request-10": '{"problem": "Extracted Problem", "method": "Extracted Method"}'
    }

    df_merged = parse_and_merge_results(df, results)

    # Check merging format: original_val \n abstract: \n snippet
    assert "Existing Problem\nabstract:\nExtracted Problem" == df_merged["problem"].iloc[0]
    assert "Existing Method\nabstract:\nExtracted Method" == df_merged["method"].iloc[0]


def test_abstract_filtering():
    """Test the abstract filtering logic."""
    from edel.pipeline.structuring import filter_abstracts

    df = pd.DataFrame(
        [
            {"abstract_text": "Too short."},  # 2 words, 1 sentence
            {"abstract_text": "This is a longer abstract with multiple sentences. It should pass."},
            {"abstract_text": ""},
            {"abstract_text": None},
        ]
    )

    # Filter with min_sentences=2, min_tokens=5
    df_filtered, report = filter_abstracts(df, min_sentences=2, min_tokens=5)

    assert len(df_filtered) == 1
    assert "It should pass" in df_filtered["abstract_text"].iloc[0]


def test_structuring_stage_batch_chunking(run_config, tmp_path):
    """Test Stage 2 in batch mode with small batch_size to trigger chunking."""
    run_config["processing_mode"] = "batch"
    run_config["structured_abstracts"]["batch_size"] = 2  # Small size to force 3 chunks for 5 docs

    # 1. Setup Stage 1 dependency
    df_stage1 = ensure_stage_1_artifacts(run_config, tmp_path)
    assert len(df_stage1) == 5

    # 2. Run Stage 2
    df_stage2, report = run_structuring_stage(df_stage1, run_config)

    # 3. Verify results
    assert isinstance(df_stage2, pd.DataFrame)
    assert len(df_stage2) == 5
    assert df_stage2["problem"].str.contains("Mock").all()


def test_llm_client_factory():
    """Test the LLM client factory with different providers."""
    from edel.io.llm import LMStudioClient, MockClient, OpenAIClient, get_llm_client

    # Mock
    client = get_llm_client({"provider": "mock"})
    assert isinstance(client, MockClient)

    # LM Studio
    client = get_llm_client({"provider": "lmstudio", "model": "test-model"})
    assert isinstance(client, LMStudioClient)
    assert client.model == "test-model"

    # OpenAI
    client = get_llm_client({"provider": "openai"})
    assert isinstance(client, OpenAIClient)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_structuring_openai_integration(tmp_path):
    """Integration test with real OpenAI (requires API key)."""
    config = {
        "data": {
            "provider": {
                "type": "scigen_null",
                "topic_id": "TSCIGEN",
                "topic_name": "SCIGen Test",
                "params": {"n_documents": 5},
            }
        },
        "structured_abstracts": {
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        "processing_mode": "simple",
    }

    df_stage1 = ensure_stage_1_artifacts(config, tmp_path)
    df_stage2, report = run_structuring_stage(df_stage1, config)

    assert len(df_stage2) == 5
    # Verify that we got some non-mock content
    assert df_stage2["problem"].str.len().mean() > 5


def test_structuring_lmstudio_integration(tmp_path):
    """Integration test with LM Studio (requires local server)."""
    import httpx

    config = {
        "data": {
            "provider": {
                "type": "scigen_null",
                "topic_id": "TSCIGEN",
                "topic_name": "SCIGen Test",
                "params": {"n_documents": 5},
            }
        },
        "structured_abstracts": {
            "provider": "lmstudio",
            "model": "gemma-4-e4b-it", # User asked for gemma-4-e4b-it
        },
        "processing_mode": "simple",
    }

    # Check if LM Studio is running
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    try:
        httpx.get(base_url.replace("/v1", "/"))
    except Exception:
        pytest.skip("LM Studio server not found at " + base_url)

    df_stage1 = ensure_stage_1_artifacts(config, tmp_path)
    df_stage2, report = run_structuring_stage(df_stage1, config)

    assert len(df_stage2) == 5


def test_structuring_stage_null(run_config, tmp_path):
    """Test Stage 2 using the null slicer provider."""
    # 1. Update config to use null provider
    run_config["structured_abstracts"]["provider"] = "null"
    run_config["structured_abstracts"]["model"] = "null-model"

    # 2. Setup Stage 1 dependency
    df_stage1 = ensure_stage_1_artifacts(run_config, tmp_path)
    assert len(df_stage1) == 5

    # Force a known abstract text for testing slicing precisely, and clear other aspects
    test_abstract = "A B C D " * 20  # Length 160, 80 tokens
    df_stage1.loc[df_stage1.index[0], "abstract_text"] = test_abstract
    for col in ["problem", "method", "finding", "interpretation"]:
        df_stage1.loc[df_stage1.index[0], col] = ""

    # 3. Run Stage 2
    df_stage2, report = run_structuring_stage(df_stage1, run_config)

    # 4. Verify results
    assert isinstance(df_stage2, pd.DataFrame)
    assert len(df_stage2) == 5
    
    first_row = df_stage2.iloc[0]
    
    # Check that abstract is sliced into four equal parts of length 40
    # Merging logic adds "abstract:\n" and strips the snippets.
    expected_snippet = ("A B C D " * 5).strip()
    assert first_row["problem"] == f"abstract:\n{expected_snippet}"
    assert first_row["method"] == f"abstract:\n{expected_snippet}"
    assert first_row["finding"] == f"abstract:\n{expected_snippet}"
    assert first_row["interpretation"] == f"abstract:\n{expected_snippet}"


def test_structuring_stage_none(run_config, tmp_path):
    """Test Stage 2 using the none provider."""
    # 1. Update config to use none provider
    run_config["structured_abstracts"]["provider"] = "none"

    # 2. Setup Stage 1 dependency
    df_stage1 = ensure_stage_1_artifacts(run_config, tmp_path)
    assert len(df_stage1) == 5

    # If the required columns are missing, it should raise ValueError
    df_missing = df_stage1.drop(columns=["problem"])
    with pytest.raises(ValueError) as exc_info:
        run_structuring_stage(df_missing, run_config)
    assert "required column 'problem' is missing" in str(exc_info.value)

    # Now add the columns
    df_stage1["problem"] = "Stage 1 Problem"
    df_stage1["method"] = "Stage 1 Method"
    df_stage1["finding"] = "Stage 1 Finding"
    df_stage1["interpretation"] = "Stage 1 Interpretation"

    # Run Stage 2
    df_stage2, report = run_structuring_stage(df_stage1, run_config)

    # Verify results are unchanged and kept exactly as they came from Stage 1
    assert isinstance(df_stage2, pd.DataFrame)
    assert len(df_stage2) == 5

    first_row = df_stage2.iloc[0]
    assert first_row["problem"] == "Stage 1 Problem"
    assert first_row["method"] == "Stage 1 Method"
    assert first_row["finding"] == "Stage 1 Finding"
    assert first_row["interpretation"] == "Stage 1 Interpretation"

    # Verify that metrics are included in the report
    assert "len_problem" in report
    assert report["initial_count"] == 5
    assert report["final_count"] == 5

    # Test sampling when n_documents is configured
    run_config["structured_abstracts"]["n_documents"] = 2
    df_stage2_sampled, report_sampled = run_structuring_stage(df_stage1, run_config)
    assert len(df_stage2_sampled) == 2
    assert report_sampled["sampled_count"] == 2
    assert report_sampled["final_count"] == 2
