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


def _split_on_top_level_implies(prop: str) -> list[str]:
    """Split a proposition string on top-level ==> or ⟹ operators, respecting parenthesis nesting."""
    parts = []
    current = []
    depth = 0
    i = 0
    n = len(prop)
    while i < n:
        char = prop[i]
        if char in "([{‹":
            depth += 1
            current.append(char)
            i += 1
        elif char in ")]}›":
            depth = max(0, depth - 1)
            current.append(char)
            i += 1
        elif depth == 0 and (prop[i:i+3] == "⟹" or prop[i:i+3] == "==>"):
            parts.append("".join(current).strip())
            current = []
            i += 3
        else:
            current.append(char)
            i += 1
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def _parse_premises_and_conclusion(statement: str) -> tuple[str, str]:
    """Parse a lemma/theorem statement to extract its premises and conclusion."""
    # Normalise whitespace
    stmt = re.sub(r'\s+', ' ', statement).strip()
    
    # 1. Handle Isar assumes/shows form
    if "shows" in stmt:
        assumes_part = stmt[:stmt.find("shows")]
        shows_part = stmt[stmt.find("shows") + len("shows"):]
        
        # Extract assumptions inside quotes
        assumptions = re.findall(r'\bassumes\s+"([^"]+)"', assumes_part)
        if not assumptions:
            assumptions = re.findall(r'"([^"]+)"', assumes_part)
            
        conclusion_match = re.search(r'"([^"]+)"', shows_part)
        conclusion = conclusion_match.group(1) if conclusion_match else shows_part.strip()
        
        premises = ", ".join(assumptions) if assumptions else "none"
        return premises, conclusion
        
    # 2. Standard forms: lemma foo: "A ==> B" or lemma "A ==> B"
    # Find content of outermost quotes
    m = re.search(r'"([^"]+)"', stmt)
    if not m:
        # Fallback: if no quotes, strip keyword/name and parse
        clean = re.sub(
            r'^(?:lemma|theorem|corollary|proposition|schematic_goal)\s+'
            r'(?:[a-zA-Z0-9_\'\.]+\s*(?:\[[^\]]*\])?\s*:)?\s*',
            '',
            stmt
        )
        prop = clean
    else:
        prop = m.group(1)
        
    parts = _split_on_top_level_implies(prop)
    if len(parts) > 1:
        premises = ", ".join(parts[:-1])
        conclusion = parts[-1]
    else:
        premises = "none"
        conclusion = parts[0]
        
    return premises, conclusion


def extract_aspects(
    lemma: dict[str, Any],
    theory_header: str = "",          # kept for backward compatibility; unused
    entry_metadata: dict[str, Any] | None = None,  # kept for backward compatibility; unused
    text_comments: list[str] | None = None,
) -> dict[str, str]:
    """Extract source-enriched aspects from a parsed lemma.

    Aspects:
        aspect_statement    — Premises/hypotheses (or "none" if unconditional).
        aspect_context      — Conclusion / consequent.
        aspect_strategy     — Proof skeleton (declarative have/show/also/case segments).
        aspect_dependencies  — Operational tactics/automation (apply/by/using segments).
    """
    statement = lemma.get("statement_text", "").strip()
    proof = lemma.get("proof_text", "").strip()
    comments = text_comments or lemma.get("text_comments", [])

    # 1. Parse statement into premises and conclusion
    premises, conclusion = _parse_premises_and_conclusion(statement)
    aspect_statement = premises
    aspect_context = conclusion

    # 2. Extract skeleton and tactics
    skeleton_segs = list(lemma.get("skeleton_segments", []))
    tactic_segs = list(lemma.get("tactic_segments", []))

    # Fallback if segments are missing but proof is present
    if not skeleton_segs and not tactic_segs and proof:
        ISAR_SKELETON_KEYWORDS = {
            "proof", "qed", "have", "show", "also", "finally", "next",
            "case", "assume", "fix", "obtain", "define", "let", "presume", "suppose"
        }
        for line in proof.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            tokens = set(re.findall(r'[a-zA-Z_]+', stripped))
            if tokens & ISAR_SKELETON_KEYWORDS:
                skeleton_segs.append(stripped)
            else:
                tactic_segs.append(stripped)

    # Clean the segments
    cleaned_skeleton = []
    for seg in skeleton_segs:
        cleaned = _strip_isabelle_markup(seg)
        if cleaned:
            cleaned_skeleton.append(cleaned)

    cleaned_tactics = []
    for seg in tactic_segs:
        cleaned = _strip_isabelle_markup(seg)
        if cleaned:
            cleaned_tactics.append(cleaned)

    # 3. Incorporate text comments into the skeleton aspect (method)
    comment_parts: list[str] = []
    if comments:
        for raw_comment in comments:
            cleaned = _strip_isabelle_markup(raw_comment)
            if len(cleaned) > 20:
                comment_parts.append(cleaned)

    strategy_parts = cleaned_skeleton + comment_parts
    aspect_strategy = "\n".join(strategy_parts)
    aspect_dependencies = "\n".join(cleaned_tactics)

    return {
        "aspect_statement": aspect_statement,
        "aspect_context": aspect_context,
        "aspect_strategy": aspect_strategy,
        "aspect_dependencies": aspect_dependencies,
    }
