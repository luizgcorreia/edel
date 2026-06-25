"""Unit tests for the RAG MCP server."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from edel.isabelle import rag_server

@pytest.fixture
def mock_index_and_client(monkeypatch):
    # Mock embedding client
    mock_client = MagicMock()
    mock_client.generate_embedding.return_value = [1.0, 0.0]
    monkeypatch.setattr(rag_server, "get_embedding_client", lambda: mock_client)
    
    # Configure mock index
    idx = rag_server.index
    idx.metadata = [
        {
            "title": "HOL.List.append_Nil",
            "problem": "[] @ ys = ys",
            "finding": "simp",
            "interpretation": "none",
            "theory": "HOL.List",
            "file": "List.thy",
            "line": 10,
        }
    ]
    idx.embeddings["problem"] = np.array([[1.0, 0.0]], dtype=np.float32)
    idx.embeddings["method"] = np.array([[1.0, 0.0]], dtype=np.float32)
    idx.embeddings["finding"] = np.array([[1.0, 0.0]], dtype=np.float32)
    idx.embeddings["interpretation"] = np.array([[1.0, 0.0]], dtype=np.float32)
    
    # Clean live index
    idx.live_metadata = []
    idx.live_embeddings = {
        "problem": [],
        "method": [],
        "finding": [],
        "interpretation": [],
    }
    return idx, mock_client


@pytest.mark.anyio
async def test_search_lemmas(mock_index_and_client):
    idx, _ = mock_index_and_client
    res = await rag_server.search_lemmas(query="test", aspect="statement")
    assert "HOL.List.append_Nil" in res
    assert "[] @ ys = ys" in res


@pytest.mark.anyio
async def test_search_strategies(mock_index_and_client):
    idx, _ = mock_index_and_client
    res = await rag_server.search_strategies(goal="test")
    assert "simp" in res
    assert "HOL.List.append_Nil" in res


@pytest.mark.anyio
async def test_related_lemmas(mock_index_and_client):
    idx, _ = mock_index_and_client
    # Let's add a second lemma so related_lemmas has something to return besides itself
    idx.metadata.append({
        "title": "HOL.List.append_Cons",
        "problem": "(x # xs) @ ys = x # (xs @ ys)",
        "finding": "simp",
        "interpretation": "none",
        "theory": "HOL.List",
        "file": "List.thy",
        "line": 20,
    })
    idx.embeddings["problem"] = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    
    res = await rag_server.related_lemmas(lemma_name="HOL.List.append_Nil")
    assert "HOL.List.append_Cons" in res


@pytest.mark.anyio
async def test_store_and_session_lemmas(mock_index_and_client):
    idx, client = mock_index_and_client
    
    res = await rag_server.store_lemma(
        name="my_new_lemma",
        statement="A ==> A",
        proof_text="by simp",
        theory="MyTheory"
    )
    assert "Successfully stored" in res
    
    # Check session lemmas
    res_list = await rag_server.session_lemmas()
    assert "my_new_lemma" in res_list
    assert "A ==> A" in res_list


@pytest.mark.anyio
async def test_persist_session_lemmas(mock_index_and_client, tmp_path, monkeypatch):
    idx, client = mock_index_and_client
    
    # Set INDEX_DIR to a temp directory
    monkeypatch.setattr(rag_server, "INDEX_DIR", str(tmp_path / "rag_index_persisted"))
    
    # 1. Try to persist when empty
    res_empty = await rag_server.persist_session_lemmas()
    assert "No new session lemmas to persist" in res_empty
    
    # 2. Store a lemma
    await rag_server.store_lemma(
        name="my_new_lemma",
        statement="A ==> A",
        proof_text="by simp",
        theory="MyTheory"
    )
    
    # 3. Persist and check success
    res_persist = await rag_server.persist_session_lemmas()
    assert "Successfully persisted 1 session lemmas" in res_persist
    
    # Verify index files were created in temp dir
    assert (tmp_path / "rag_index_persisted" / "metadata.parquet").exists()
    assert (tmp_path / "rag_index_persisted" / "embeddings.npz").exists()

