import pytest
import pandas as pd
from edel.providers.lexicon_null import generate_dataset as generate_lexicon_null
from edel.providers.syntax_null import generate_dataset as generate_syntax_null
from edel.providers.scigen_null import generate_dataset as generate_scigen_null
from edel.providers.openalex import generate_dataset as generate_openalex
from edel.providers.afp import generate_dataset as generate_afp
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
    
    df, _ = generate_lexicon_null(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    
    # Check that abstract and title are strings and not empty
    assert all(isinstance(val, str) for val in df["abstract_text"])
    assert all(len(val) > 0 for val in df["abstract_text"])
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
    
    df1, _ = generate_lexicon_null(config)
    df2, _ = generate_lexicon_null(config)
    
    pd.testing.assert_frame_equal(df1, df2)

def test_run_data_stage_lexicon_null(base_run_config):
    """Test the Stage 1 orchestrator with lexicon_null provider."""
    # Modify config to use lexicon_null
    config = base_run_config.copy()
    config["data"]["provider"]["type"] = "lexicon_null"
    config["data"]["provider"]["params"]["n_documents"] = 3
    
    df, _ = run_data_stage(config["data"])
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


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
    
    df, _ = generate_syntax_null(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    
    assert all(isinstance(val, str) for val in df["abstract_text"])
    assert all(len(val) > 0 for val in df["abstract_text"])


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
    
    df1, _ = generate_syntax_null(config)
    df2, _ = generate_syntax_null(config)
    
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
    
    df, _ = generate_scigen_null(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    
    assert "abstract_text" in df.columns
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
    
    df, _ = generate_openalex(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 10  # Might be less if topic is small or filters are strict
    if len(df) > 0:
        for col in REQUIRED_COLUMNS:
            assert col in df.columns
        
        assert "abstract_text" in df.columns
        assert isinstance(df["abstract_text"].iloc[0], str)
        assert len(df["abstract_text"].iloc[0]) > 0


def test_afp_provider_direct():
    """Test the afp provider using the local repository mirror."""
    config = {
        "provider": {
            "type": "afp",
            "repo_url": "https://foss.heptapod.net/isa-afp/afp-2025-2",
            "params": {
                "n_documents": 5
            }
        }
    }
    
    df, _ = generate_afp(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    
    assert "abstract_text" in df.columns
    assert "title" in df.columns
    assert all(df["source_provider"].str.startswith("afp"))
    # Check that some semantic aspects were filled
    assert any(df["method"] != "")
    assert any(df["finding"] != "")


def test_openalex_proportional_temporal_mocked():
    """Test proportional_temporal sampling strategy with mocked API responses."""
    from unittest.mock import patch
    
    config = {
        "random_seed": 100,
        "provider": {
            "type": "openalex",
            "topic_id": "T12345",
            "params": {
                "sampling_strategy": "proportional_temporal",
                "sample_percentage": "5%"
            }
        }
    }
    
    mock_group_by = {
        "group_by": [
            {"key": "2020", "count": 100},
            {"key": "2021", "count": 10},
        ]
    }

    def side_effect(filters, cursor=None, group_by=None, sample=None, seed=None, sort=None, per_page=200):
        if group_by == "publication_year":
            return mock_group_by
        elif "publication_year:2020" in filters:
            assert sample == 5
            assert seed == 100  # root random_seed is injected
            # return 5 unique works
            return {
                "results": [
                    {"id": f"W2020_{i}", "title": f"Paper {i}", "publication_year": 2020, "abstract_inverted_index": {"hello": [0]}}
                    for i in range(5)
                ]
            }
        elif "publication_year:2021" in filters:
            assert sample == 1  # max(1, round(10 * 0.05)) = 1
            assert seed == 100
            return {
                "results": [
                    {"id": "W2021_0", "title": "Paper 2021", "publication_year": 2021, "abstract_inverted_index": {"world": [0]}}
                ]
            }
        return {"results": []}

    with patch("edel.providers.openalex.openalex_request", side_effect=side_effect) as mock_req:
        df, report = generate_openalex(config)
        
        # Verify the target count:
        # Year 2020: 100 * 0.05 = 5
        # Year 2021: 10 * 0.05 = 0.5 -> round to 0 -> max(1, 0) = 1
        # Total target = 6
        assert len(df) == 6
        assert len(df[df["publication_year"] == 2020]) == 5
        assert len(df[df["publication_year"] == 2021]) == 1
        
        # Check call parameters
        assert mock_req.call_count == 3  # 1 group_by + 2 year queries


def test_openalex_percentage_normalization():
    """Test that all percentage config formats are correctly parsed and normalized."""
    from unittest.mock import patch
    
    mock_group_by = {"group_by": [{"key": "2020", "count": 100}]}
    
    formats = ["5%", 0.05, 5, "0.05", "5"]
    
    for fmt in formats:
        config = {
            "provider": {
                "type": "openalex",
                "topic_id": "T12345",
                "params": {
                    "sampling_strategy": "proportional_temporal",
                    "sample_percentage": fmt
                }
            }
        }
        
        def side_effect(filters, cursor=None, group_by=None, sample=None, seed=None, sort=None, per_page=200):
            if group_by == "publication_year":
                return mock_group_by
            # For 5%, target is 5
            assert sample == 5
            return {"results": [{"id": f"W_{i}", "title": "T", "publication_year": 2020, "abstract_inverted_index": {"h": [0]}} for i in range(5)]}
            
        with patch("edel.providers.openalex.openalex_request", side_effect=side_effect):
            df, _ = generate_openalex(config)
            assert len(df) == 5


def test_openalex_deterministic_proportional_temporal_mocked():
    """Test proportional_temporal_deterministic strategy with mocked responses."""
    from unittest.mock import patch
    
    config = {
        "provider": {
            "type": "openalex",
            "topic_id": "T12345",
            "params": {
                "sampling_strategy": "proportional_temporal_deterministic",
                "sample_percentage": 0.05
            }
        }
    }
    
    mock_group_by = {
        "group_by": [
            {"key": "2020", "count": 100}
        ]
    }
    
    def side_effect(filters, cursor=None, group_by=None, sample=None, seed=None, sort=None, per_page=200):
        if group_by == "publication_year":
            return mock_group_by
        
        assert cursor is not None
        assert sort == "cited_by_count:desc"
        assert per_page == 5
        return {
            "results": [
                {"id": f"W_{i}", "title": f"Paper {i}", "publication_year": 2020, "abstract_inverted_index": {"hello": [0]}}
                for i in range(5)
            ]
        }
        
    with patch("edel.providers.openalex.openalex_request", side_effect=side_effect):
        df, _ = generate_openalex(config)
        assert len(df) == 5


def test_afp_rag_provider(tmp_path):
    """Test the afp_rag provider using a temporary RAG index."""
    import numpy as np
    from edel.isabelle.index import NumpyRAGIndex
    from edel.providers.afp_rag import generate_dataset as generate_afp_rag
    
    # 1. Create a dummy index with 3 lemmas
    index = NumpyRAGIndex()
    index.metadata = [
        {
            "title": "Session1.Theory1.lemma_a",
            "problem": "lemma_a statement",
            "method": "lemma_a context",
            "finding": "lemma_a strategy",
            "interpretation": "lemma_b", # lemma_a references lemma_b
            "theory": "Session1.Theory1",
            "file": "Theory1.thy",
            "line": 10,
            "proof_text": "by simp",
            "statement_text": "lemma lemma_a"
        },
        {
            "title": "Session1.Theory1.lemma_b",
            "problem": "lemma_b statement",
            "method": "lemma_b context",
            "finding": "lemma_b strategy",
            "interpretation": "lemma_c", # lemma_b references lemma_c
            "theory": "Session1.Theory1",
            "file": "Theory1.thy",
            "line": 20,
            "proof_text": "by simp",
            "statement_text": "lemma lemma_b"
        },
        {
            "title": "Session2.Theory2.lemma_c",
            "problem": "lemma_c statement",
            "method": "lemma_c context",
            "finding": "lemma_c strategy",
            "interpretation": "none",
            "theory": "Session2.Theory2",
            "file": "Theory2.thy",
            "line": 30,
            "proof_text": "by simp",
            "statement_text": "lemma lemma_c"
        }
    ]
    # Set dummy embeddings (dim=1536)
    for aspect in ["problem", "method", "finding", "interpretation"]:
        index.embeddings[aspect] = np.random.rand(3, 1536).astype(np.float32)
        
    # Save dummy index
    index_dir = tmp_path / "dummy_index"
    index.save(index_dir)
    
    # 2. Run data provider
    config = {
        "provider": {
            "type": "afp_rag",
            "params": {
                "index_dir": index_dir
            }
        }
    }
    
    df, _ = generate_afp_rag(config)
    
    # 3. Assertions
    assert len(df) == 3
    assert "problem_embedding" in df.columns
    assert "method_embedding" in df.columns
    assert "finding_embedding" in df.columns
    assert "interpretation_embedding" in df.columns
    
    # Check citation counts (cited_by_count)
    # lemma_a is cited 0 times
    # lemma_b is cited 1 time (by lemma_a)
    # lemma_c is cited 1 time (by lemma_b)
    citation_dict = dict(zip(df["title"], df["cited_by_count"]))
    assert citation_dict["Session1.Theory1.lemma_a"] == 0
    assert citation_dict["Session1.Theory1.lemma_b"] == 1
    assert citation_dict["Session2.Theory2.lemma_c"] == 1
