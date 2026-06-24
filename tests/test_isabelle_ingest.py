"""Unit tests for the Isabelle ingestion module."""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from edel.isabelle import ingest

def test_ingest_session_lemmas(monkeypatch):
    # Mock AFPMetadataParser to return dummy metadata
    class MockMetadataParser:
        def load_entry_metadata(self, entry_name):
            return {
                "title": "Mock Session",
                "abstract": "Mock Abstract",
                "topics": ["Mock/Topic"]
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
                "   2  lemma test_lemma [simp]: \"A ==> A\"\n"
                "   4    by simp\n"
            )
        elif 'Ir.source_map "MockSession.TestTheory"' in ml_command:
            return (
                "   0  theory                   1       0  TestTheory.thy\n"
                "   2  lemma                    2      39  TestTheory.thy\n"
                "   4  by                       3      75  TestTheory.thy\n"
            )
        return ""
        
    mock_client.send.side_effect = mock_send
    
    # Replace client instantiation
    monkeypatch.setattr(ingest, "EphemeralReplClient", lambda host, port, token: mock_client)
    
    df = ingest.ingest_session_lemmas(token="dummy-token")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["title"].iloc[0] == "MockSession.TestTheory.test_lemma"
    assert df["problem"].iloc[0] == "A ==> A"
    assert "Mock Abstract" in df["method"].iloc[0]
    assert df["finding"].iloc[0] == "simp"
    assert df["interpretation"].iloc[0] == "none"
    assert df["theory"].iloc[0] == "MockSession.TestTheory"
    assert df["file"].iloc[0] == "TestTheory.thy"
