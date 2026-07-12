"""Unit tests for the Isabelle ingestion module."""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from edel.il import ingest

def test_ingest_session_lemmas(monkeypatch):
    # Mock AFPMetadataParser to return dummy metadata
    class MockMetadataParser:
        def load_entry_metadata(self, entry_name):
            return {
                "title": "Mock Session",
                "abstract": "Mock Abstract",
                "topics": ["Mock/Topic"],
                "date": "2024-05-15"
            }
    monkeypatch.setattr(ingest, "AFPMetadataParser", MockMetadataParser)
    
    # Mock EphemeralReplClient
    mock_client = MagicMock()
    
    # Configure responses
    def mock_send(ml_command):
        if "Ir.theories" in ml_command:
            return "MockSession.TestTheory\n"
        elif 'Ir.source "MockSession.TestTheory"' in ml_command:
            return (
                "   0  theory TestTheory imports Main begin\n"
                "   2  definition my_definition where \"my_definition x = x\"\n"
                "   4  lemma test_lemma [simp]: \"my_definition A = A\"\n"
                "   6    by (simp add: my_definition_def)\n"
                "            "
            )
        elif 'Ir.source_map "MockSession.TestTheory"' in ml_command:
            return (
                "   0  theory                   1       0  TestTheory.thy\n"
                "   2  definition               2      39  TestTheory.thy\n"
                "   4  lemma                    3      90  TestTheory.thy\n"
                "   6  by                       4     130  TestTheory.thy\n"
            )
        return ""
        
    mock_client.send.side_effect = mock_send
    
    # Replace client instantiation
    monkeypatch.setattr(ingest, "EphemeralReplClient", lambda host, port, token: mock_client)
    
    df = ingest.ingest_session_lemmas(token="dummy-token")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    
    # Check definition
    df_def = df[df["title"].str.contains("my_definition")]
    assert len(df_def) == 1
    # Treated as 0-simplex, all aspects are equal to the definition statement
    stmt = 'definition my_definition where "my_definition x = x"'
    assert df_def["finding"].iloc[0] == stmt
    assert df_def["interpretation"].iloc[0] == stmt
    # dependents holds dependents for definitions
    assert "MockSession.TestTheory.test_lemma" in df_def["dependents"].iloc[0]
    assert df_def["publication_year"].iloc[0] == 2024
    
    # Check lemma
    df_lemma = df[df["title"].str.contains("test_lemma")]
    assert len(df_lemma) == 1
    # no split -> falls back to statement
    assert df_lemma["problem"].iloc[0] == "my_definition A = A"
    # interpretation is conclusion
    assert df_lemma["interpretation"].iloc[0] == "my_definition A = A"
    # method is skeleton (tactic-only proof collapses to tactics)
    assert "by (simp add: my_definition_def)" in df_lemma["method"].iloc[0]
    # finding is tactics
    assert "by (simp add: my_definition_def)" in df_lemma["finding"].iloc[0]
    assert df_lemma["theory"].iloc[0] == "MockSession.TestTheory"
    assert df_lemma["file"].iloc[0] == "TestTheory.thy"
    assert df_lemma["publication_year"].iloc[0] == 2024


