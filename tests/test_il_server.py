"""Unit tests for the RAG MCP server."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from edel.il import il_server

@pytest.fixture
def mock_index_and_client(monkeypatch):
    # Mock embedding client
    mock_client = MagicMock()
    mock_client.generate_embedding.return_value = [1.0, 0.0]
    monkeypatch.setattr(il_server, "get_embedding_client", lambda: mock_client)
    
    # Configure mock index
    idx = il_server.index
    idx.metadata = [
        {
            "title": "HOL.List.append_Nil",
            "problem": "none",
            "method": "",
            "finding": "by simp",
            "interpretation": "[] @ ys = ys",
            "theory": "HOL.List",
            "file": "List.thy",
            "line": 10,
            "keyword": "lemma",
            "cited_deps": "none",
            "dependents": "none"
        }
    ]
    idx.embeddings["problem"] = np.array([[1.0, 0.0]], dtype=np.float32)
    idx.embeddings["method"] = np.array([[1.0, 0.0]], dtype=np.float32)
    idx.embeddings["finding"] = np.array([[1.0, 0.0]], dtype=np.float32)
    idx.embeddings["interpretation"] = np.array([[1.0, 0.0]], dtype=np.float32)

    idx.definition_metadata = []
    idx.definition_embeddings = None
    
    # Clean live index
    idx.live_metadata = []
    idx.live_embeddings = {
        "problem": [],
        "method": [],
        "finding": [],
        "interpretation": [],
    }
    idx.live_definition_metadata = []
    idx.live_definition_embeddings = []
    return idx, mock_client


@pytest.mark.anyio
async def test_search_lemmas(mock_index_and_client):
    idx, _ = mock_index_and_client
    res = await il_server.search_lemmas(query="test", aspect="conclusion")
    assert "HOL.List.append_Nil" in res
    assert "[] @ ys = ys" in res


@pytest.mark.anyio
async def test_search_definitions(mock_index_and_client):
    idx, _ = mock_index_and_client
    idx.definition_metadata = [{
        "title": "HOL.List.my_def",
        "problem": "my_def x = x + 1",
        "method": "",
        "finding": "",
        "interpretation": "",
        "theory": "HOL.List",
        "keyword": "definition",
        "dependents": "none"
    }]
    idx.definition_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    
    res = await il_server.search_definitions(query="my_def")
    assert "HOL.List.my_def" in res
    assert "my_def x = x + 1" in res


@pytest.mark.anyio
async def test_related_lemmas(mock_index_and_client):
    idx, _ = mock_index_and_client
    # Let's add a second lemma so related_lemmas has something to return besides itself
    idx.metadata.append({
        "title": "HOL.List.append_Cons",
        "problem": "none",
        "method": "",
        "finding": "by simp",
        "interpretation": "(x # xs) @ ys = x # (xs @ ys)",
        "theory": "HOL.List",
        "file": "List.thy",
        "line": 20,
        "keyword": "lemma"
    })
    idx.embeddings["interpretation"] = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    
    res = await il_server.related_lemmas(lemma_name="HOL.List.append_Nil")
    assert "HOL.List.append_Cons" in res


@pytest.mark.anyio
async def test_store_and_session_lemmas(mock_index_and_client):
    idx, client = mock_index_and_client
    
    res = await il_server.store_lemma(
        name="my_new_lemma",
        statement="A ==> A",
        proof_text="by simp",
        theory="MyTheory"
    )
    assert "Successfully stored" in res
    
    # Store a definition too
    res_def = await il_server.store_definition(
        name="my_new_def",
        statement="my_new_def x = x",
        theory="MyTheory"
    )
    assert "Successfully stored definition" in res_def
    
    # Check session items
    res_list = await il_server.session_lemmas()
    assert "my_new_lemma" in res_list
    assert "my_new_def" in res_list
    assert "Conclusion" in res_list


@pytest.mark.anyio
async def test_persist_session_lemmas(mock_index_and_client, tmp_path, monkeypatch):
    idx, client = mock_index_and_client
    
    # Set INDEX_DIR to a temp directory
    monkeypatch.setattr(il_server, "INDEX_DIR", str(tmp_path / "rag_index_persisted"))
    
    # 1. Try to persist when empty
    res_empty = await il_server.persist_session_lemmas()
    assert "No new session items to persist" in res_empty
    
    # 2. Store a lemma and definition
    await il_server.store_lemma(
        name="my_new_lemma",
        statement="A ==> A",
        proof_text="by simp",
        theory="MyTheory"
    )
    await il_server.store_definition(
        name="my_new_def",
        statement="my_new_def x = x",
        theory="MyTheory"
    )
    
    # 3. Persist and check success
    res_persist = await il_server.persist_session_lemmas()
    assert "Successfully persisted 1 session lemmas and 1 session definitions" in res_persist
    
    # Verify index files were created in temp dir
    assert (tmp_path / "rag_index_persisted" / "metadata.parquet").exists()
    assert (tmp_path / "rag_index_persisted" / "embeddings.npz").exists()
    assert (tmp_path / "rag_index_persisted" / "definitions_metadata.parquet").exists()
    assert (tmp_path / "rag_index_persisted" / "definitions_embeddings.npz").exists()


@pytest.mark.anyio
async def test_il_proof_strategy_prompt(mock_index_and_client):
    res = il_server.il_proof_strategy()
    assert "Isabelle/Isar assistant" in res
    assert "I/L (Isabelle/Landscape)" in res
    assert "premises" in res
    assert "search_definitions" in res
    assert "dependents" in res

