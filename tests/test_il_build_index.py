"""Unit tests for the build_index script."""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from edel.il import build_il_index

def test_build_index_main(tmp_path, monkeypatch):
    # Mocking ingest_session_lemmas
    mock_df = pd.DataFrame([
        {
            "title": "HOL.List.append_Nil",
            "problem": "none",
            "method": "",
            "finding": "by simp",
            "interpretation": "[] @ ys = ys",
            "theory": "HOL.List",
            "file": "List.thy",
            "line": 10,
        }
    ])
    
    # Mock run_embedding_stage to return df with embedding columns
    mock_df_embedded = mock_df.copy()
    mock_df_embedded["problem_embedding"] = json.dumps([1.0, 0.0])
    mock_df_embedded["method_embedding"] = json.dumps([0.0, 1.0])
    mock_df_embedded["finding_embedding"] = json.dumps([0.5, 0.5])
    mock_df_embedded["interpretation_embedding"] = json.dumps([0.1, 0.9])
    
    # Mocking functions
    monkeypatch.setattr(build_il_index, "ingest_session_lemmas", lambda **kwargs: mock_df)
    monkeypatch.setattr(build_il_index, "run_embedding_stage", lambda df, config: mock_df_embedded)
    
    output_dir = tmp_path / "test_rag_index"
    
    # Run build_index with mocked args
    test_args = [
        "build_il_index.py",
        "--token", "dummy-token",
        "--output", str(output_dir),
        "--provider", "openai",
        "--model", "test-model"
    ]
    
    with patch("sys.argv", test_args):
        build_il_index.main()
        
    # Check that output files were created
    assert (output_dir / "metadata.parquet").exists()
    assert (output_dir / "embeddings.npz").exists()


def test_build_index_skip_embedding(tmp_path, monkeypatch):
    # Mocking ingest_session_lemmas
    mock_df = pd.DataFrame([
        {
            "title": "HOL.List.append_Nil",
            "problem": "none",
            "method": "",
            "finding": "by simp",
            "interpretation": "[] @ ys = ys",
            "theory": "HOL.List",
            "file": "List.thy",
            "line": 10,
            "keyword": "lemma"
        }
    ])
    
    # Mocking functions
    monkeypatch.setattr(build_il_index, "ingest_session_lemmas", lambda **kwargs: mock_df)
    
    output_dir = tmp_path / "test_rag_index_skip"
    
    # Run build_index with mocked args
    test_args = [
        "build_il_index.py",
        "--token", "dummy-token",
        "--output", str(output_dir),
        "--skip-embedding"
    ]
    
    with patch("sys.argv", test_args):
        build_il_index.main()
        
    # Check that output files were created
    assert (output_dir / "metadata.parquet").exists()
    assert (output_dir / "embeddings.npz").exists()
    
    # Load index to verify it works without embeddings
    from edel.il.index import NumpyRAGIndex
    idx = NumpyRAGIndex()
    idx.load(output_dir)
    assert len(idx.metadata) == 1
    assert idx.embeddings["problem"] is None


def test_aspect_parsing_rules():
    """Verify the four-rule cascade in _parse_premises_and_conclusion."""
    from edel.il.aspects import _parse_premises_and_conclusion as parse

    # ── Rule 1: Conditional ───────────────────────────────────────────────────
    # 1a. Standard "A ⟹ B"
    p, c = parse('lemma foo: "A ⟹ B"')
    assert p == "A" and c == "B"

    # 1b. assumes / shows
    p, c = parse('lemma foo: assumes "A" shows "B"')
    assert p == "A" and c == "B"

    # 1c. shows "A ⟹ B" — previously broken, now fixed
    p, c = parse('lemma sq: fixes e :: real shows "e > 0 ⟹ ∃d. 0 < d"')
    assert p == "e > 0" and c == "∃d. 0 < d"

    # 1d. ⟦A; B⟧ ⟹ C bracket form
    p, c = parse('lemma foo: "⟦A; B⟧ ⟹ C"')
    assert p == "A, B" and c == "C"

    # 1e. Multiple premises: A ⟹ B ⟹ C
    p, c = parse('lemma foo: "x ∈ s ⟹ y ∈ s ⟹ x = y"')
    assert "x ∈ s" in p and "y ∈ s" in p
    assert c == "x = y"

    # ── Rule 2: Equivalence ───────────────────────────────────────────────────
    # 2a. Strict iff ⟷
    p, c = parse('lemma foo: "(P ∧ Q) ⟷ (Q ∧ P)"')
    assert p == "(P ∧ Q)" and c == "(Q ∧ P)"

    # 2b. Equality as rewrite — both sides complex
    p, c = parse('lemma ball_insert: "(∀x∈insert a B. P x) = (P a ∧ (∀x∈B. P x))"')
    assert "insert a B" in p and "P a" in c

    # 2c. Equality — simple LHS: should NOT split (fixed point instead)
    p, c = parse('lemma foo: "n = card S"')
    assert p == c  # fixed point

    # 2d. Simple equality both sides — fixed point
    p, c = parse('lemma foo: "x = y"')
    assert p == c

    # ── Rule 3: Obtains ───────────────────────────────────────────────────────
    # 3a. assumes + obtains
    p, c = parse('lemma foo: assumes "open S" obtains x where "x ∈ S"')
    assert p == "open S" and "x ∈ S" in c

    # 3b. Pure obtains (no assumes) — fixed point
    p, c = parse('lemma foo: obtains x where "0 < x"')
    assert p == c  # fixed point

    # ── Rule 4: Unconditional ─────────────────────────────────────────────────
    # 4a. Symmetric rewrite — both sides complex → rule 2b splits correctly
    p, c = parse('lemma foo: "dist x y = dist y x"')
    assert p == "dist x y" and c == "dist y x"

    # 4b. Absolute value rewrite — both sides complex → rule 2b splits
    p, c = parse('lemma real_abs_dist: "¦dist x y¦ = dist x y"')
    assert "dist x y" in p and c == "dist x y"

    # 4c. True fixed-point: simple variable binding (LHS is plain name)
    p, c = parse('lemma foo: "n = card S"')
    assert p == c  # n is simple → falls to rule 4


def test_simplex_collapse_rules():
    """Verify simplex collapse in extract_aspects across all proof structural types."""
    from edel.il.aspects import extract_aspects

    def make_lemma(statement, proof="", skeleton=None, tactics=None):
        return {
            "statement_text": statement,
            "proof_text": proof,
            "skeleton_segments": skeleton or [],
            "tactic_segments": tactics or [],
            "text_comments": [],
        }

    # ── Full Isar (3-simplex): skeleton ≠ tactics ──────────────────────────
    lemma = make_lemma(
        'lemma foo: "A ⟹ B"',
        proof='proof\n  have "A" by auto\n  then show "B" by blast\nqed',
        skeleton=["proof", "have A", "then show B", "qed"],
        tactics=["by auto", "by blast"],
    )
    r = extract_aspects(lemma)
    assert r["aspect_strategy"] != r["aspect_dependencies"], "Full Isar: M should ≠ F"
    assert r["aspect_statement"] != r["aspect_context"], "Full Isar: P should ≠ I"

    # ── Rule M1: tactic-only (2-simplex, M=F) ──────────────────────────────
    lemma = make_lemma(
        'lemma foo: "A ⟹ B"',
        proof="by simp",
        skeleton=[],
        tactics=["by simp"],
    )
    r = extract_aspects(lemma)
    assert r["aspect_strategy"] == r["aspect_dependencies"] == "by simp", \
        "Rule M1: tactic-only proof should collapse M=F=tactics"

    # ── Rule M2: skeleton-only (2-simplex, M=F) ────────────────────────────
    lemma = make_lemma(
        'lemma foo: "A ⟹ B"',
        proof="proof\n  show B by done\nqed",
        skeleton=["proof", "show B", "qed"],
        tactics=[],
    )
    r = extract_aspects(lemma)
    assert r["aspect_strategy"] == r["aspect_dependencies"], \
        "Rule M2: skeleton-only proof should collapse M=F=skeleton"
    assert "proof" in r["aspect_strategy"]

    # ── Rule M3: no proof content (M=F=I collapse) ─────────────────────────
    lemma = make_lemma(
        'lemma foo: "A ⟹ B"',
        proof="",
        skeleton=[],
        tactics=[],
    )
    r = extract_aspects(lemma)
    assert r["aspect_strategy"] == r["aspect_dependencies"] == r["aspect_context"], \
        "Rule M3: no proof → M=F=interpretation"

    # ── 0-simplex: axiomatic unconditional (P=M=F=I) ───────────────────────
    lemma = make_lemma(
        'lemma foo: "dist x y = dist y x"',  # rule 2b splits: P=LHS, I=RHS
        proof="",
        skeleton=[],
        tactics=[],
    )
    r = extract_aspects(lemma)
    # P ≠ I (equivalence split), but M=F=I (rule M3)
    assert r["aspect_strategy"] == r["aspect_dependencies"] == r["aspect_context"]

    # ── 0-simplex: truly axiomatic (P=I unconditional + no proof) ───────────
    lemma = make_lemma(
        'lemma foo: "x = y"',  # simple equality: P=I (rule 4 fixed-point)
        proof="",
        skeleton=[],
        tactics=[],
    )
    r = extract_aspects(lemma)
    assert r["aspect_statement"] == r["aspect_context"], "Unconditional: P=I"
    assert r["aspect_strategy"] == r["aspect_dependencies"] == r["aspect_context"], \
        "0-simplex: P=M=F=I"
    # All four coincide
    vals = {r["aspect_statement"], r["aspect_context"],
            r["aspect_strategy"], r["aspect_dependencies"]}
    assert len(vals) == 1, f"0-simplex: expected 1 unique value, got {len(vals)}: {vals}"


