import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from edel.il.index import NumpyRAGIndex
from edel.il.compute_landscape_height import compute_and_save_landscape_height


def test_landscape_height_computation_and_search():
    # 1. Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # 2. Build mock lemma metadata
        # A depends on B, B depends on C (so A transitively depends on C)
        # B dependents: A (1)
        # C dependents: B, A (2)
        # A dependents: none (0)
        lemmas_data = [
            {
                "title": "Test.lemma_A",
                "problem": "premises_A",
                "method": "skeleton_A",
                "finding": "tactics_A",
                "interpretation": "conclusion_A",
                "theory": "Test",
                "keyword": "lemma",
                "file": "test.thy",
                "line": 10,
                "proof_text": "proof A using Test.lemma_B qed",
                "statement_text": "lemma A",
                "cited_deps": "Test.lemma_B",
                "dependents": "none",
            },
            {
                "title": "Test.lemma_B",
                "problem": "premises_B",
                "method": "skeleton_B",
                "finding": "tactics_B",
                "interpretation": "conclusion_B",
                "theory": "Test",
                "keyword": "lemma",
                "file": "test.thy",
                "line": 20,
                "proof_text": "proof B using Test.lemma_C qed",
                "statement_text": "lemma B",
                "cited_deps": "Test.lemma_C",
                "dependents": "none",
            },
            {
                "title": "Test.lemma_C",
                "problem": "premises_C",
                "method": "skeleton_C",
                "finding": "tactics_C",
                "interpretation": "conclusion_C",
                "theory": "Test",
                "keyword": "lemma",
                "file": "test.thy",
                "line": 30,
                "proof_text": "proof C qed",
                "statement_text": "lemma C",
                "cited_deps": "none",
                "dependents": "none",
            },
        ]
        
        # D is a definition, used in Lemma A
        defs_data = [
            {
                "title": "Test.def_D",
                "problem": "definition D",
                "method": "",
                "finding": "",
                "interpretation": "",
                "theory": "Test",
                "keyword": "definition",
                "file": "test.thy",
                "line": 5,
                "proof_text": "",
                "statement_text": "definition D",
                "cited_deps": "none",
                "dependents": "Test.lemma_A",
            }
        ]
        
        # 3. Save as parquet files
        pd.DataFrame(lemmas_data).to_parquet(temp_dir / "metadata.parquet", index=False)
        pd.DataFrame(defs_data).to_parquet(temp_dir / "definitions_metadata.parquet", index=False)
        
        # Save mock embeddings (dimension 4)
        npz_kwargs = {
            "problem": np.random.randn(3, 4).astype(np.float32),
            "method": np.random.randn(3, 4).astype(np.float32),
            "finding": np.random.randn(3, 4).astype(np.float32),
            "interpretation": np.random.randn(3, 4).astype(np.float32),
        }
        np.savez_compressed(temp_dir / "embeddings.npz", **npz_kwargs)
        np.savez_compressed(temp_dir / "definitions_embeddings.npz", definitions=np.random.randn(1, 4).astype(np.float32))
        
        # 4. Run the post-processing script
        compute_and_save_landscape_height(temp_dir)
        
        # 5. Load back parquet data to verify counts
        df_lemmas = pd.read_parquet(temp_dir / "metadata.parquet")
        df_defs = pd.read_parquet(temp_dir / "definitions_metadata.parquet")
        
        counts = df_lemmas.set_index("title")["dependents_count"].to_dict()
        def_counts = df_defs.set_index("title")["dependents_count"].to_dict()
        
        assert counts["Test.lemma_C"] == 2  # B and A depend on C
        assert counts["Test.lemma_B"] == 1  # A depends on B
        assert counts["Test.lemma_A"] == 0  # No dependents
        assert def_counts["Test.def_D"] == 1  # A depends on D
        
        # 6. Test RAG index search re-ranking and filtering
        index = NumpyRAGIndex()
        index.load(temp_dir)
        
        # Search with min_dependents filter
        query_vec = [1.0, 0.0, 0.0, 0.0]
        hits_filtered = index.search(query_vec, aspect="problem", min_dependents=2)
        assert len(hits_filtered) == 1
        assert hits_filtered[0]["lemma"]["title"] == "Test.lemma_C"
        
        # Search with significance re-ranking
        # Mock index.embeddings["problem"] to have identical cosine similarities to verify rank swap
        index.embeddings["problem"] = np.array([
            [1.0, 0.0, 0.0, 0.0],  # A (0 dependents)
            [1.0, 0.0, 0.0, 0.0],  # B (1 dependent)
            [1.0, 0.0, 0.0, 0.0],  # C (2 dependents)
        ], dtype=np.float32)
        
        hits_significance = index.search([1.0, 0.0, 0.0, 0.0], aspect="problem", sort_by_significance=True)
        # All have similarity score 1.0. With sort_by_significance, C should have highest, then B, then A
        assert hits_significance[0]["lemma"]["title"] == "Test.lemma_C"
        assert hits_significance[1]["lemma"]["title"] == "Test.lemma_B"
        assert hits_significance[2]["lemma"]["title"] == "Test.lemma_A"
        
    finally:
        shutil.rmtree(temp_dir)
