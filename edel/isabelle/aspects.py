"""Aspect extraction module for Isabelle lemmas (Format B: source-enriched)."""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Tactic keyword vocabulary — lines containing these tokens belong to method
# ---------------------------------------------------------------------------
_TACTIC_KEYWORDS = {
    "apply", "by", "using", "unfolding", "proof", "qed", "done",
    "simp", "simp_all", "auto", "blast", "fastforce", "force", "metis",
    "induction", "induct", "coinduction", "cases", "case", "rule", "subst",
    "clarify", "clarsimp", "safe", "linarith", "arith", "presburger",
    "ring", "algebra", "have", "show", "obtain", "assume", "fix",
    "define", "let", "note", "then", "thus", "hence", "next",
    "defer", "prefer", "sorry", "oops", "intro", "elim", "fact",
}

# Keywords that introduce a cited identifier (dependency extraction).
# We match each introducer keyword followed by any mix of identifiers,
# brackets, and whitespace — terminated by a closing paren, semicolon, or
# another tactic keyword.  Parenthesised forms like "by (metis a b)" are
# handled by stripping punctuation from the captured text.
_DEP_INTRODUCERS = re.compile(
    r'\b(?:using|unfolding|fact|rule|subst|metis|blast|insert)\s+'
    r'([\w\'.\[\] ,\-]+)',
    re.MULTILINE,
)
_SIMP_DEP_PATTERN = re.compile(
    r'\bsimp(?:\s+(?:add|del|only))?:\s*([\w\'., \[\]]+)',
)

# Identifiers that are tactics/common keywords — excluded from deps
_DEP_EXCLUSIONS = _TACTIC_KEYWORDS | {
    "of", "where", "in", "and", "or", "not", "if", "then", "else",
    "true", "false", "add", "del", "only", "intro", "elim", "dest",
    "no_types", "full_types", "standard", "this", "that", "goal",
}

# Isabelle construct keywords for interpretation aspect
_CONSTRUCT_KEYWORDS = {
    "lemma", "theorem", "corollary", "proposition", "schematic_goal",
    "definition", "fun", "primrec", "function", "datatype", "type_synonym",
    "inductive", "coinductive", "record", "abbreviation",
}


def _strip_isabelle_markup(text: str) -> str:
    """Remove LaTeX commands and Isabelle antiquotations from text-block prose."""
    # @{term ...}, @{thm ...}, @{const ...} etc.
    text = re.sub(r'@\{[^}]*\}', '', text)
    # \cmd{...} LaTeX macros
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    # $$...$$  or  $...$  math environments
    text = re.sub(r'\$\$[^$]+\$\$', ' ', text)
    text = re.sub(r'\$[^$\n]+\$', ' ', text)
    # bare \command sequences
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # Isabelle term delimiters ‹...›
    text = re.sub(r'[‹›]', '"', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_tactic_lines(proof: str) -> str:
    """Return the subset of proof lines that contain at least one tactic keyword."""
    if not proof.strip():
        return ""
    lines = proof.splitlines()
    tactic_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = set(re.findall(r'[a-zA-Z_]+', stripped))
        if tokens & _TACTIC_KEYWORDS:
            tactic_lines.append(stripped)
    # Fallback: if we somehow found nothing, return the full proof (short proofs
    # may consist entirely of symbolic expressions like ‹...›)
    return "\n".join(tactic_lines) if tactic_lines else proof.strip()


def _extract_dependencies(proof: str) -> str:
    """Extract cited dependency identifiers from the proof body."""
    if not proof.strip():
        return "none"

    deps: set[str] = set()

    # Patterns: using/unfolding/fact/rule/subst/metis/blast/insert X Y Z
    for m in _DEP_INTRODUCERS.finditer(proof):
        # Strip surrounding punctuation (parens, brackets, etc.) before splitting
        raw = re.sub(r'[();\[\]]', ' ', m.group(1))
        for word in re.findall(r'[a-zA-Z][a-zA-Z0-9_\'.]+', raw):
            if word.lower() not in _DEP_EXCLUSIONS and len(word) > 2:
                deps.add(word)

    # simp add: / simp only: / simp del:
    for m in _SIMP_DEP_PATTERN.finditer(proof):
        raw = re.sub(r'[();\[\]]', ' ', m.group(1))
        for word in re.findall(r'[a-zA-Z][a-zA-Z0-9_\'.]+', raw):
            if word.lower() not in _DEP_EXCLUSIONS and len(word) > 2:
                deps.add(word)

    return ", ".join(sorted(deps)) if deps else "none"


def extract_aspects(
    lemma: dict[str, Any],
    theory_header: str = "",          # kept for backward compatibility; unused
    entry_metadata: dict[str, Any] | None = None,  # kept for backward compatibility; unused
    text_comments: list[str] | None = None,
) -> dict[str, str]:
    """Extract source-enriched Format B aspects from a parsed lemma.

    Aspects:
        aspect_statement   — Full ``statement_text`` verbatim (keyword + name + proposition).
        aspect_context     — Isabelle construct keyword + qualified theory name.
        aspect_strategy    — Proof tactic/structural lines, plus cleaned text-block commentary.
        aspect_dependencies — Cited dependency identifiers extracted from the proof.

    Args:
        lemma: Parsed lemma dict with keys ``statement_text``, ``proof_text``,
               ``theory``, ``keyword``.
        theory_header: Legacy parameter; no longer used.
        entry_metadata: Legacy parameter; no longer used.
        text_comments: Optional list of raw text-block segment contents adjacent
                       to this lemma (e.g. from ``text ‹...›`` commands).
    """
    statement = lemma.get("statement_text", "").strip()
    proof = lemma.get("proof_text", "").strip()
    theory = lemma.get("theory", "unknown")
    keyword = lemma.get("keyword", "")

    # ------------------------------------------------------------------
    # 1. aspect_statement: full statement_text verbatim
    # ------------------------------------------------------------------
    aspect_statement = statement

    # ------------------------------------------------------------------
    # 2. aspect_context: construct keyword + qualified theory name
    # ------------------------------------------------------------------
    # Determine the keyword from the statement if not supplied in the dict
    if not keyword:
        kw_match = re.match(
            r'^(lemma|theorem|corollary|proposition|schematic_goal|'
            r'definition|fun|function|primrec|datatype|type_synonym|'
            r'inductive|coinductive|record|abbreviation)\b',
            statement,
        )
        keyword = kw_match.group(1) if kw_match else "unknown"

    # Normalise: strip internal qualifiers from the keyword string
    kw_label = keyword if keyword in _CONSTRUCT_KEYWORDS else keyword.split()[0]
    aspect_context = f"{kw_label} in {theory}"

    # ------------------------------------------------------------------
    # 3. aspect_strategy: tactic lines + cleaned text-block commentary
    # ------------------------------------------------------------------
    tactic_part = _extract_tactic_lines(proof)

    comment_parts: list[str] = []
    if text_comments:
        for raw_comment in text_comments:
            cleaned = _strip_isabelle_markup(raw_comment)
            # Only include if there's meaningful prose left after stripping
            if len(cleaned) > 20:
                comment_parts.append(cleaned)

    strategy_parts = [p for p in [tactic_part] + comment_parts if p]
    aspect_strategy = "\n".join(strategy_parts)

    # ------------------------------------------------------------------
    # 4. aspect_dependencies: cited dependency identifiers
    # ------------------------------------------------------------------
    aspect_dependencies = _extract_dependencies(proof)

    return {
        "aspect_statement": aspect_statement,
        "aspect_context": aspect_context,
        "aspect_strategy": aspect_strategy,
        "aspect_dependencies": aspect_dependencies,
    }
