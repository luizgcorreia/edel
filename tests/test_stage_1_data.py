import pytest
import pandas as pd
from edel.providers.lexicon_null import generate_dataset as generate_lexicon_null
from edel.providers.syntax_null import generate_dataset as generate_syntax_null
from edel.providers.scigen_null import generate_dataset as generate_scigen_null
from edel.providers.openalex import generate_dataset as generate_openalex
from edel.pipeline.data import run_data_stage
from edel.providers.base import REQUIRED_COLUMNS

def test_lexicon_null_provider_direct():
    """Test the lexicon_null provider directly."""
    config = {
        "provider": {
            "type": "lexicon_null",
            "params": {
                "n_documents": 10,
                "seed": 42
            }
        }
    }
    
    df = generate_lexicon_null(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    
    # Check that abstract and title are strings and not empty
    assert all(isinstance(val, str) for val in df["abstract"])
    assert all(len(val) > 0 for val in df["abstract"])
    assert all(isinstance(val, str) for val in df["title"])
    assert all(len(val) > 0 for val in df["title"])

def test_lexicon_null_determinism():
    """Test that lexicon_null is deterministic with a seed."""
    config = {
        "provider": {
            "type": "lexicon_null",
            "params": {
                "n_documents": 5,
                "seed": 42
            }
        }
    }
    
    df1 = generate_lexicon_null(config)
    df2 = generate_lexicon_null(config)
    
    pd.testing.assert_frame_equal(df1, df2)

def test_run_data_stage_lexicon_null(base_run_config, tmp_path):
    """Test the Stage 1 orchestrator with lexicon_null provider."""
    # Modify config to use lexicon_null
    config = base_run_config.copy()
    config["data"]["provider"]["type"] = "lexicon_null"
    config["data"]["provider"]["params"]["n_documents"] = 3
    
    artifacts = run_data_stage(config["data"], tmp_path)
    
    assert "dataset" in artifacts
    art = artifacts["dataset"]
    
    # Check path structure: artifacts/data_collection/<label>/dataset__<hash>.parquet
    assert "data_collection" in str(art.parquet_path)
    # Label should be lexicon_null_unknown_global (because topic_id is not in config["data"]["provider"] top-level but inside data section?)
    # Wait, get_experiment_label uses data_cfg = config.get("data", config).
    # In run_data_stage, we pass config["data"].
    assert art.label == "lexicon_null_T10102_global"
    
    assert art.parquet_path.exists() or art.pkl_path.exists()
    
    # Load and check
    from edel.io.artifact import load_artifact
    df = load_artifact(art)
    assert len(df) == 3


def test_syntax_null_provider_direct():
    """Test the syntax_null provider directly."""
    config = {
        "provider": {
            "type": "syntax_null",
            "params": {
                "n_documents": 5,
                "seed": 42
            }
        }
    }
    
    df = generate_syntax_null(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    
    assert all(isinstance(val, str) for val in df["abstract"])
    assert all(len(val) > 0 for val in df["abstract"])


def test_syntax_null_determinism():
    """Test that syntax_null is deterministic with a seed."""
    config = {
        "provider": {
            "type": "syntax_null",
            "params": {
                "n_documents": 3,
                "seed": 123
            }
        }
    }
    
    df1 = generate_syntax_null(config)
    df2 = generate_syntax_null(config)
    
    pd.testing.assert_frame_equal(df1, df2)


def test_scigen_null_provider_direct():
    """Test the scigen_null provider (clones repo if needed)."""
    config = {
        "provider": {
            "type": "scigen_null",
            "params": {
                "n_documents": 5
            }
        }
    }
    
    df = generate_scigen_null(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    
    assert "abstract" in df.columns
    assert "title" in df.columns


def test_openalex_provider_direct():
    """Test the openalex provider with a real API call (limited to 10 docs)."""
    config = {
        "provider": {
            "type": "openalex",
            "topic_id": "T10102",
            "params": {
                "n_documents": 10
            }
        }
    }
    
    df = generate_openalex(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 10  # Might be less if topic is small or filters are strict
    if len(df) > 0:
        for col in REQUIRED_COLUMNS:
            assert col in df.columns
        
        assert "abstract" in df.columns
        assert isinstance(df["abstract"].iloc[0], str)
        assert len(df["abstract"].iloc[0]) > 0
