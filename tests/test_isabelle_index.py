"""Unit tests for the Numpy vector index."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from edel.isabelle.index import NumpyRAGIndex

def test_numpy_rag_index_lifecycle(tmp_path):
    # 1. Create a dummy dataframe with embeddings as JSON strings
    df = pd.DataFrame([
        {
            "title": "HOL.List.append_Nil",
            "problem": "[] @ ys = ys",
            "method": "Theory List",
            "finding": "simp",
            "interpretation": "none",
            "theory": "HOL.List",
            "problem_embedding": json.dumps([1.0, 0.0]),
            "method_embedding": json.dumps([0.0, 1.0]),
        },
        {
            "title": "HOL.List.append_Cons",
            "problem": "(x # xs) @ ys = x # (xs @ ys)",
            "method": "Theory List",
            "finding": "simp",
            "interpretation": "none",
            "theory": "HOL.List",
            "problem_embedding": json.dumps([0.8, 0.6]),
            "method_embedding": json.dumps([0.0, 1.0]),
        }
    ])
    
    # 2. Build index
    idx = NumpyRAGIndex()
    idx.build_from_dataframe(df)
    
    assert len(idx.metadata) == 2
    assert idx.embeddings["problem"].shape == (2, 2)
    assert np.allclose(idx.embeddings["problem"][0], [1.0, 0.0])
    
    # 3. Save index
    save_dir = tmp_path / "rag_index"
    idx.save(save_dir)
    assert (save_dir / "metadata.parquet").exists()
    assert (save_dir / "embeddings.npz").exists()
    
    # 4. Load index into a new object
    idx2 = NumpyRAGIndex()
    idx2.load(save_dir)
    assert len(idx2.metadata) == 2
    assert idx2.embeddings["problem"].shape == (2, 2)
    
    # 5. Search index
    results = idx2.search(query_vector=[1.0, 0.0], aspect="problem", max_results=5)
    assert len(results) == 2
    assert results[0]["lemma"]["title"] == "HOL.List.append_Nil"
    assert results[0]["score"] == pytest.approx(1.0)
    
    # Near match
    results = idx2.search(query_vector=[0.8, 0.6], aspect="problem", max_results=5)
    assert results[0]["lemma"]["title"] == "HOL.List.append_Cons"
    assert results[0]["score"] == pytest.approx(1.0)
    
    # 6. Add live session lemma
    idx2.add_live_lemma(
        name="my_new_lemma",
        aspect_text_dict={"problem": "new proposition", "method": "my theory"},
        embeddings_dict={"problem": [0.6, 0.8], "method": [1.0, 0.0]},
        theory="MyTheory"
    )
    
    assert len(idx2.live_metadata) == 1
    
    # Search including live lemma
    results = idx2.search(query_vector=[0.6, 0.8], aspect="problem", max_results=5)
    assert len(results) == 3
    assert results[0]["lemma"]["title"] == "MyTheory.my_new_lemma"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["source"] == "live"
    
    # Test theory filtering
    results_filtered = idx2.search(query_vector=[0.6, 0.8], aspect="problem", max_results=5, theory_filter="HOL")
    assert len(results_filtered) == 2
    assert all("HOL" in r["lemma"]["theory"] for r in results_filtered)

    # 7. Test persistence of live lemmas
    persist_dir = tmp_path / "rag_index_persisted"
    idx2.persist_live_lemmas(persist_dir)
    
    # Verify live metadata/embeddings are cleared in memory
    assert len(idx2.live_metadata) == 0
    assert len(idx2.live_embeddings["problem"]) == 0
    
    # Verify static metadata/embeddings now include the new lemma (total 3 lemmas)
    assert len(idx2.metadata) == 3
    assert idx2.embeddings["problem"].shape == (3, 2)
    assert idx2.metadata[-1]["title"] == "MyTheory.my_new_lemma"
    
    # Load from disk and verify it has all 3 lemmas
    idx3 = NumpyRAGIndex()
    idx3.load(persist_dir)
    assert len(idx3.metadata) == 3
    assert idx3.embeddings["problem"].shape == (3, 2)
    assert idx3.metadata[-1]["title"] == "MyTheory.my_new_lemma"

