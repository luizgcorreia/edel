"""Unit tests for Isabelle aspect extraction (Format B: source-enriched)."""

from edel.il.aspects import extract_aspects, _strip_isabelle_markup


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

    # problem: premises of the statement (none for unconditional)
    assert aspects["aspect_statement"] == 'none'

    # method: declarative skeleton (empty for simple proof)
    assert aspects["aspect_strategy"] == ""

    # finding: tactics
    assert "by simp" in aspects["aspect_dependencies"]

    # interpretation: conclusion of the statement
    assert aspects["aspect_context"] == '[] @ ys = ys'


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

    # problem: premises of statement (none for shows without assumptions)
    assert aspects["aspect_statement"] == 'none'

    # method: declarative skeleton (empty as there are no declare statements)
    assert aspects["aspect_strategy"] == ""

    # finding: tactics
    assert "apply (induction xs)" in aspects["aspect_dependencies"]
    assert "by (metis lemma2)" in aspects["aspect_dependencies"]

    # interpretation: conclusion
    assert aspects["aspect_context"] == "mset (xs @ ys) = mset xs + mset ys"


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

    # problem: premises (none)
    assert aspects["aspect_statement"] == 'none'

    # method: empty
    assert aspects["aspect_strategy"] == ""

    # finding: empty tactics
    assert aspects["aspect_dependencies"] == ""

    # interpretation: definition body (conclusion)
    assert aspects["aspect_context"] == 'my_def x = x + 1'


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
    assert aspects["aspect_dependencies"] == ""
    assert aspects["aspect_context"] == "nat ⇒ bool"
    assert aspects["aspect_statement"] == "none"


# ---------------------------------------------------------------------------
# Text comments are cleaned and appended to method (tactics)
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

    # The tactic line must be present in tactics (aspect_dependencies)
    assert "by (rule bar)" in aspects["aspect_dependencies"]

    # The cleaned long comment must be appended/prepended to skeleton (aspect_strategy)
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
    # interpretation comes from conclusion, NOT from entry_metadata
    assert aspects["aspect_context"] == "True"
    assert "Foo Entry" not in aspects["aspect_context"]
