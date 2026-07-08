"""Unit tests for Isabelle aspect extraction (Format B: source-enriched)."""

from edel.isabelle.aspects import extract_aspects, _strip_isabelle_markup


# ---------------------------------------------------------------------------
# Markup stripping helper
# ---------------------------------------------------------------------------

def test_strip_isabelle_markup_removes_antiquotations():
    raw = "We use @{thm foo} and @{term bar} here."
    assert "@{" not in _strip_isabelle_markup(raw)
    assert "We use" in _strip_isabelle_markup(raw)


def test_strip_isabelle_markup_removes_latex():
    raw = r"This uses \emph{induction} and $x^2$ math."
    cleaned = _strip_isabelle_markup(raw)
    assert "\\" not in cleaned
    assert "$" not in cleaned


# ---------------------------------------------------------------------------
# Basic lemma (simple proof)
# ---------------------------------------------------------------------------

def test_extract_aspects_basic():
    lemma = {
        "statement_text": 'lemma append_Nil [simp]: "[] @ ys = ys"',
        "proof_text": "by simp",
        "theory": "HOL.List",
        "keyword": "lemma"
    }

    aspects = extract_aspects(lemma)

    # problem: full statement_text verbatim
    assert aspects["aspect_statement"] == 'lemma append_Nil [simp]: "[] @ ys = ys"'

    # method: tactic line(s) from the proof
    assert "by simp" in aspects["aspect_strategy"]

    # finding: no explicit citations in "by simp"
    assert aspects["aspect_dependencies"] == "none"

    # interpretation: keyword + theory
    assert aspects["aspect_context"] == "lemma in HOL.List"


# ---------------------------------------------------------------------------
# Complex lemma (apply-style + induction + citations)
# ---------------------------------------------------------------------------

def test_extract_aspects_complex():
    lemma = {
        "statement_text": 'theorem my_complex_thm:\n  shows "mset (xs @ ys) = mset xs + mset ys"',
        "proof_text": (
            "apply (induction xs)\n"
            "apply auto\n"
            "using other_lemma [simp] by (metis lemma2)"
        ),
        "theory": "HOL.Multiset",
        "keyword": "theorem"
    }

    aspects = extract_aspects(lemma)

    # problem: full statement verbatim
    assert 'theorem my_complex_thm' in aspects["aspect_statement"]
    assert 'mset (xs @ ys) = mset xs + mset ys' in aspects["aspect_statement"]

    # method: apply, induction, auto, metis lines all present
    assert "apply" in aspects["aspect_strategy"]
    assert "induction" in aspects["aspect_strategy"]

    # finding: other_lemma and lemma2 extracted as citations
    deps = [d.strip() for d in aspects["aspect_dependencies"].split(",")]
    assert "other_lemma" in deps
    assert "lemma2" in deps

    # interpretation
    assert aspects["aspect_context"] == "theorem in HOL.Multiset"


# ---------------------------------------------------------------------------
# Definition (no proof body)
# ---------------------------------------------------------------------------

def test_extract_aspects_definition():
    defn = {
        "statement_text": 'definition my_def where "my_def x = x + 1"',
        "proof_text": "",
        "theory": "HOL.List",
        "keyword": "definition"
    }

    aspects = extract_aspects(defn)

    # problem: full definition text verbatim
    assert aspects["aspect_statement"] == 'definition my_def where "my_def x = x + 1"'

    # method: empty — no proof body
    assert aspects["aspect_strategy"] == ""

    # finding: no citations
    assert aspects["aspect_dependencies"] == "none"

    # interpretation: keyword + theory
    assert aspects["aspect_context"] == "definition in HOL.List"


# ---------------------------------------------------------------------------
# Inductive definition
# ---------------------------------------------------------------------------

def test_extract_aspects_inductive():
    defn = {
        "statement_text": 'inductive even :: "nat ⇒ bool" where\n  zero: "even 0" |\n  step: "even n ⟹ even (Suc (Suc n))"',
        "proof_text": "",
        "theory": "HOL.Nat",
        "keyword": "inductive"
    }
    aspects = extract_aspects(defn)

    assert aspects["aspect_strategy"] == ""
    assert aspects["aspect_context"] == "inductive in HOL.Nat"
    assert "inductive even" in aspects["aspect_statement"]


# ---------------------------------------------------------------------------
# Text comments are cleaned and appended to method
# ---------------------------------------------------------------------------

def test_extract_aspects_text_comments_cleaned_and_appended():
    lemma = {
        "statement_text": 'lemma foo: "P ⟹ Q"',
        "proof_text": "by (rule bar)",
        "theory": "HOL.Foo",
        "keyword": "lemma"
    }
    raw_comments = [
        # LaTeX and antiquotation noise mixed with real explanation
        r"We proceed by @{thm bar} using \emph{rule} application.",
        "Short.",  # too short — should be excluded (< 20 chars after stripping)
    ]

    aspects = extract_aspects(lemma, text_comments=raw_comments)

    # The tactic line must be present
    assert "by (rule bar)" in aspects["aspect_strategy"]

    # The cleaned long comment must be appended
    assert "We proceed by" in aspects["aspect_strategy"]
    assert "@{" not in aspects["aspect_strategy"]  # antiquotation stripped

    # The short comment must be excluded
    assert "Short." not in aspects["aspect_strategy"]


# ---------------------------------------------------------------------------
# Dependency extraction: simp add: and using patterns
# ---------------------------------------------------------------------------

def test_extract_aspects_simp_dependencies():
    lemma = {
        "statement_text": 'lemma size_empty [simp]: "size {#} = 0"',
        "proof_text": "by (simp add: size_multiset_overloaded_def)",
        "theory": "HOL-Library.Multiset",
        "keyword": "lemma"
    }
    aspects = extract_aspects(lemma)
    assert "size_multiset_overloaded_def" in aspects["aspect_dependencies"]


def test_extract_aspects_using_dependencies():
    lemma = {
        "statement_text": 'lemma multiset_eq_iff: "M = N ⟷ (∀a. count M a = count N a)"',
        "proof_text": "by (simp only: count_inject [symmetric] fun_eq_iff)",
        "theory": "HOL-Library.Multiset",
        "keyword": "lemma"
    }
    aspects = extract_aspects(lemma)
    deps = aspects["aspect_dependencies"]
    assert "count_inject" in deps or "fun_eq_iff" in deps


# ---------------------------------------------------------------------------
# Backward compatibility: legacy params are ignored gracefully
# ---------------------------------------------------------------------------

def test_extract_aspects_legacy_params_ignored():
    """theory_header and entry_metadata are accepted but no longer used."""
    lemma = {
        "statement_text": 'lemma foo: "True"',
        "proof_text": "by simp",
        "theory": "HOL.Foo",
        "keyword": "lemma"
    }
    # Should not raise
    aspects = extract_aspects(
        lemma,
        theory_header="theory Foo imports Main begin",
        entry_metadata={"title": "Foo Entry", "abstract": "Stuff"},
    )
    # interpretation comes from keyword + theory, NOT from entry_metadata
    assert aspects["aspect_context"] == "lemma in HOL.Foo"
    assert "Foo Entry" not in aspects["aspect_context"]
