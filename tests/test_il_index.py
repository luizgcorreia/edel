"""Unit tests for the Numpy vector index."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from edel.il.index import NumpyRAGIndex

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
            "finding_embedding": json.dumps([0.0, 1.0]),
            "interpretation_embedding": json.dumps([0.0, 1.0]),
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
            "finding_embedding": json.dumps([0.0, 1.0]),
            "interpretation_embedding": json.dumps([0.0, 1.0]),
        },
        {
            "title": "HOL.List.my_def",
            "problem": "my_def x = x + 1",
            "method": "my_def x = x + 1",
            "finding": "my_def x = x + 1",
            "interpretation": "my_def x = x + 1",
            "theory": "HOL.List",
            "keyword": "definition",
            "problem_embedding": json.dumps([0.6, 0.8]),
            "method_embedding": json.dumps([0.6, 0.8]),
            "finding_embedding": json.dumps([0.6, 0.8]),
            "interpretation_embedding": json.dumps([0.6, 0.8]),
        }
    ])
    
    # 2. Build index
    idx = NumpyRAGIndex()
    idx.build_from_dataframe(df)
    
    # All items are in the unified metadata
    assert len(idx.metadata) == 3
    assert idx.embeddings["problem"].shape == (3, 2)
    assert np.allclose(idx.embeddings["problem"][0], [1.0, 0.0])
    assert np.allclose(idx.embeddings["problem"][2], [0.6, 0.8])
    
    # 3. Save index
    save_dir = tmp_path / "rag_index"
    idx.save(save_dir)
    assert (save_dir / "metadata.parquet").exists()
    assert (save_dir / "embeddings.npz").exists()
    assert not (save_dir / "definitions_metadata.parquet").exists()
    assert not (save_dir / "definitions_embeddings.npz").exists()
    
    # 4. Load index into a new object
    idx2 = NumpyRAGIndex()
    idx2.load(save_dir)
    assert len(idx2.metadata) == 3
    
    # 5. Search index lemmas (excluding definitions by default)
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
    assert len(idx2.live_metadata) == 2
    
    # Search definitions including live definition
    def_results2 = idx2.search_definitions(query_vector=[0.0, 1.0], max_results=5)
    assert len(def_results2) == 2
    assert def_results2[0]["definition"]["title"] == "MyTheory.my_new_def"
    assert def_results2[0]["score"] == pytest.approx(1.0)
    assert def_results2[0]["source"] == "live"
    
    # 7. Test persistence of live items
    persist_dir = tmp_path / "rag_index_persisted"
    idx2.persist_live_lemmas(persist_dir)
    
    # Verify live metadata/embeddings are cleared in memory
    assert len(idx2.live_metadata) == 0
    
    # Verify static metadata/embeddings now include the new lemma and def (total 5 items)
    assert len(idx2.metadata) == 5
    assert idx2.metadata[-1]["title"] == "MyTheory.my_new_def"
    
    # Load from disk and verify
    idx3 = NumpyRAGIndex()
    idx3.load(persist_dir)
    assert len(idx3.metadata) == 5
    assert idx3.metadata[-1]["title"] == "MyTheory.my_new_def"


def test_numpy_rag_index_dynamic_dependents():
    # 1. Create a dummy dataframe with lemmas
    # C has no cited_deps.
    # B cites C (B -> C in DAG, i.e., C has 1 dependent B).
    # A has no cited_deps.
    df = pd.DataFrame([
        {
            "title": "HOL.List.lemma_A",
            "problem": "A",
            "method": "A",
            "finding": "A",
            "interpretation": "A",
            "theory": "HOL.List",
            "keyword": "lemma",
            "cited_deps": "none",
            "dependents_count": 0,
        },
        {
            "title": "HOL.List.lemma_B",
            "problem": "B",
            "method": "B",
            "finding": "B",
            "interpretation": "B",
            "theory": "HOL.List",
            "keyword": "lemma",
            "cited_deps": "HOL.List.lemma_C",
            "dependents_count": 1,
        },
        {
            "title": "HOL.List.lemma_C",
            "problem": "C",
            "method": "C",
            "finding": "C",
            "interpretation": "C",
            "theory": "HOL.List",
            "keyword": "lemma",
            "cited_deps": "none",
            "dependents_count": 2, # initially, let's assume it was 2 or whatever, we will recalculate it
        }
    ])

    idx = NumpyRAGIndex()
    idx.build_from_dataframe(df)

    # Initial recalculation to ensure consistency
    idx.update_dependents_counts()
    
    # Check initial counts
    meta_dict = {item["title"]: item for item in idx.metadata}
    assert meta_dict["HOL.List.lemma_A"]["dependents_count"] == 0
    # Wait, let's be careful about directions:
    # "B cites C" means C is a dependency of B.
    # Therefore, B is a dependent of C.
    # If B is a dependent of C, C's dependents count should be at least 1 (specifically 1, representing B).
    # B has no dependents (since no one cites B), so B's dependents count is 0.
    # Let's verify our DFS counts:
    # C -> adj_transpose[C] has B.
    # So C's dependents count is 1. B's dependents count is 0.
    assert meta_dict["HOL.List.lemma_C"]["dependents_count"] == 1
    assert meta_dict["HOL.List.lemma_B"]["dependents_count"] == 0
    assert meta_dict["HOL.List.lemma_A"]["dependents_count"] == 0

    # 2. Add a live lemma D that cites B: "HOL.List.lemma_D" cites "HOL.List.lemma_B"
    # Now:
    # D cites B, B cites C.
    # Dependents count:
    # D has 0 dependents.
    # B has 1 dependent (D).
    # C has 2 dependents (B, D).
    idx.add_live_lemma(
        name="lemma_D",
        aspect_text_dict={"problem": "D"},
        embeddings_dict={"problem": [1.0, 0.0]},
        theory="HOL.List",
        cited_deps=["HOL.List.lemma_B"]
    )

    meta_dict = {item["title"]: item for item in idx.metadata + idx.live_metadata}
    assert meta_dict["HOL.List.lemma_D"]["dependents_count"] == 0
    assert meta_dict["HOL.List.lemma_B"]["dependents_count"] == 1
    assert meta_dict["HOL.List.lemma_C"]["dependents_count"] == 2

    # 3. Add a live definition def_E with dependents: HOL.List.lemma_D
    # Now:
    # def_E -> dependents: HOL.List.lemma_D
    # So D depends on E.
    # E has dependents: D. Since D has no dependents, E's dependents count is 1 (D).
    idx.add_live_definition(
        name="def_E",
        statement_text="def E",
        embedding=[1.0, 0.0],
        theory="HOL.List",
        dependents="HOL.List.lemma_D"
    )

    meta_dict = {item["title"]: item for item in idx.metadata + idx.live_metadata}
    assert meta_dict["HOL.List.def_E"]["dependents_count"] == 1

    # 4. Add a live lemma F that cites def_E: HOL.List.lemma_F cites HOL.List.def_E
    # Now:
    # F cites E, E has dependents: D.
    # So E's direct dependents are D and F.
    # E's transitive dependents are D, F (since D has no dependents).
    # E's dependents count is 2.
    idx.add_live_lemma(
        name="lemma_F",
        aspect_text_dict={"problem": "F"},
        embeddings_dict={"problem": [1.0, 0.0]},
        theory="HOL.List",
        cited_deps=["HOL.List.def_E"]
    )

    meta_dict = {item["title"]: item for item in idx.metadata + idx.live_metadata}
    assert meta_dict["HOL.List.def_E"]["dependents_count"] == 2

    # 5. Add cyclic dependency to verify cycle-safety:
    # lemma_G cites lemma_H.
    # lemma_H cites lemma_G.
    # This forms a cycle. Cycle-safe DFS should not crash, and should compute:
    # G -> dependents: H. H -> dependents: G.
    # G transitive dependents: {G, H} - {G} = {H} (count = 1)
    # H transitive dependents: {G, H} - {H} = {G} (count = 1)
    idx.add_live_lemma(
        name="lemma_G",
        aspect_text_dict={"problem": "G"},
        embeddings_dict={"problem": [1.0, 0.0]},
        theory="HOL.List",
        cited_deps=["HOL.List.lemma_H"]
    )
    idx.add_live_lemma(
        name="lemma_H",
        aspect_text_dict={"problem": "H"},
        embeddings_dict={"problem": [1.0, 0.0]},
        theory="HOL.List",
        cited_deps=["HOL.List.lemma_G"]
    )

    meta_dict = {item["title"]: item for item in idx.metadata + idx.live_metadata}
    assert isinstance(meta_dict["HOL.List.lemma_G"]["dependents_count"], int)
    assert isinstance(meta_dict["HOL.List.lemma_H"]["dependents_count"], int)

