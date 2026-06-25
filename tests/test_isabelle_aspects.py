"""Unit tests for Isabelle aspect extraction."""

from edel.isabelle.aspects import extract_aspects

def test_extract_aspects_basic():
    lemma = {
        "statement_text": 'lemma append_Nil [simp]: "[] @ ys = ys"',
        "proof_text": "by simp",
        "theory": "HOL.List",
        "keyword": "lemma"
    }
    
    aspects = extract_aspects(lemma)
    
    assert aspects["aspect_statement"] == "[] @ ys = ys"
    assert "simp" in aspects["aspect_strategy"]
    assert "simple-style" in aspects["aspect_strategy"]
    assert "Theory: HOL.List" in aspects["aspect_context"]
    assert aspects["aspect_dependencies"] == "none"

def test_extract_aspects_complex():
    lemma = {
        "statement_text": 'theorem my_complex_thm:\n  shows "mset (xs @ ys) = mset xs + mset ys"',
        "proof_text": "apply (induction xs)\napply auto\nusing other_lemma [simp] by (metis lemma2)",
        "theory": "HOL.Multiset",
        "keyword": "theorem"
    }
    
    entry_meta = {
        "title": "Multiset Ordering",
        "abstract": "An abstract description of multisets.",
        "topics": ["Logic/Rewriting"],
    }
    
    aspects = extract_aspects(lemma, theory_header="theory Multiset imports List begin", entry_metadata=entry_meta)
    
    assert aspects["aspect_statement"] == "mset (xs @ ys) = mset xs + mset ys"
    assert "induction" in aspects["aspect_strategy"]
    assert "auto" in aspects["aspect_strategy"]
    assert "metis" in aspects["aspect_strategy"]
    assert "apply-style" in aspects["aspect_strategy"]
    
    # Check dependencies: other_lemma, lemma2
    deps = [d.strip() for d in aspects["aspect_dependencies"].split(",")]
    assert "other_lemma" in deps
    assert "lemma2" in deps
    
    assert "theory Multiset imports List begin" in aspects["aspect_context"]
    assert "Multiset Ordering" in aspects["aspect_context"]
    assert "An abstract description of multisets." in aspects["aspect_context"]


def test_extract_aspects_definition():
    defn = {
        "statement_text": 'definition my_def where "my_def x = x + 1"',
        "proof_text": "",
        "theory": "HOL.List",
        "keyword": "definition"
    }
    
    aspects = extract_aspects(defn)
    
    # Keyword is cleaned
    assert aspects["aspect_statement"] == 'my_def where "my_def x = x + 1"'
    assert "definition-style definition" in aspects["aspect_strategy"]
    assert "Theory: HOL.List" in aspects["aspect_context"]
    assert aspects["aspect_dependencies"] == "none"

