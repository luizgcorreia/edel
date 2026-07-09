"""Unit tests for the Numpy vector index."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from edel.isabelle.index import NumpyRAGIndex

def test_numpy_rag_index_lifecycle(tmp_path):
    # 1. Create a dummy dataframe with lemmas and definitions
    df = pd.DataFrame([
        {
            "title": "HOL.List.append_Nil",
            "problem": "[] @ ys = ys",
            "method": "Theory List",
            "finding": "simp",
            "interpretation": "none",
            "theory": "HOL.List",
            "keyword": "lemma",
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
            "keyword": "theorem",
            "problem_embedding": json.dumps([0.8, 0.6]),
            "method_embedding": json.dumps([0.0, 1.0]),
        },
        {
            "title": "HOL.List.my_def",
            "problem": "my_def x = x + 1",
            "method": "",
            "finding": "",
            "interpretation": "",
            "theory": "HOL.List",
            "keyword": "definition",
            "problem_embedding": json.dumps([0.6, 0.8]),
            "method_embedding": json.dumps([0.0, 1.0]),
        }
    ])
    
    # 2. Build index
    idx = NumpyRAGIndex()
    idx.build_from_dataframe(df)
    
    # Lemmas built correctly
    assert len(idx.metadata) == 2
    assert idx.embeddings["problem"].shape == (2, 2)
    assert np.allclose(idx.embeddings["problem"][0], [1.0, 0.0])
    
    # Definitions built correctly
    assert len(idx.definition_metadata) == 1
    assert idx.definition_embeddings.shape == (1, 2)
    assert np.allclose(idx.definition_embeddings[0], [0.6, 0.8])
    
    # 3. Save index
    save_dir = tmp_path / "rag_index"
    idx.save(save_dir)
    assert (save_dir / "metadata.parquet").exists()
    assert (save_dir / "embeddings.npz").exists()
    assert (save_dir / "definitions_metadata.parquet").exists()
    assert (save_dir / "definitions_embeddings.npz").exists()
    
    # 4. Load index into a new object
    idx2 = NumpyRAGIndex()
    idx2.load(save_dir)
    assert len(idx2.metadata) == 2
    assert len(idx2.definition_metadata) == 1
    assert idx2.definition_embeddings.shape == (1, 2)
    
    # 5. Search index lemmas
    results = idx2.search(query_vector=[1.0, 0.0], aspect="problem", max_results=5)
    assert len(results) == 2
    assert results[0]["lemma"]["title"] == "HOL.List.append_Nil"
    assert results[0]["score"] == pytest.approx(1.0)
    
    # Search index definitions
    def_results = idx2.search_definitions(query_vector=[0.6, 0.8], max_results=5)
    assert len(def_results) == 1
    assert def_results[0]["definition"]["title"] == "HOL.List.my_def"
    assert def_results[0]["score"] == pytest.approx(1.0)
    
    # 6. Add live session lemma
    idx2.add_live_lemma(
        name="my_new_lemma",
        aspect_text_dict={"problem": "new proposition", "method": "my theory"},
        embeddings_dict={"problem": [0.6, 0.8], "method": [1.0, 0.0]},
        theory="MyTheory"
    )
    assert len(idx2.live_metadata) == 1
    
    # Add live definition
    idx2.add_live_definition(
        name="my_new_def",
        statement_text="my_new_def x = x",
        embedding=[0.0, 1.0],
        theory="MyTheory"
    )
    assert len(idx2.live_definition_metadata) == 1
    
    # Search definitions including live definition
    def_results2 = idx2.search_definitions(query_vector=[0.0, 1.0], max_results=5)
    assert len(def_results2) == 2
    assert def_results2[0]["definition"]["title"] == "MyTheory.my_new_def"
    assert def_results2[0]["score"] == pytest.approx(1.0)
    assert def_results2[0]["source"] == "live"
    
    # 7. Test persistence of live lemmas and definitions
    persist_dir = tmp_path / "rag_index_persisted"
    idx2.persist_live_lemmas(persist_dir)
    
    # Verify live metadata/embeddings are cleared in memory
    assert len(idx2.live_metadata) == 0
    assert len(idx2.live_definition_metadata) == 0
    
    # Verify static metadata/embeddings now include the new lemma (total 3 lemmas, 2 definitions)
    assert len(idx2.metadata) == 3
    assert len(idx2.definition_metadata) == 2
    assert idx2.definition_embeddings.shape == (2, 2)
    assert idx2.definition_metadata[-1]["title"] == "MyTheory.my_new_def"
    
    # Load from disk and verify
    idx3 = NumpyRAGIndex()
    idx3.load(persist_dir)
    assert len(idx3.metadata) == 3
    assert len(idx3.definition_metadata) == 2
    assert idx3.definition_embeddings.shape == (2, 2)
    assert idx3.definition_metadata[-1]["title"] == "MyTheory.my_new_def"
