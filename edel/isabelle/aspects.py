"""Aspect extraction module for Isabelle lemmas."""

from __future__ import annotations

import re
from typing import Any


def extract_aspects(
    lemma: dict[str, Any],
    theory_header: str = "",
    entry_metadata: dict[str, Any] | None = None
) -> dict[str, str]:
    """Extract proof-oriented aspects structurally from a parsed lemma.
    
    Aspects:
        - aspect_statement: The formal theorem proposition.
        - aspect_context: Import structure, locale info, and entry abstract.
        - aspect_strategy: Classification of proof methods used.
        - aspect_dependencies: Citations of other lemmas/facts in the proof.
    """
    statement = lemma.get("statement_text", "")
    proof = lemma.get("proof_text", "")
    
    # 1. Statement aspect: extract proposition (usually inside quotes after the colon)
    stmt_clean = re.sub(r'\s+', ' ', statement.strip())
    # Check for "shows" or "obtains" first
    shows_match = re.search(r'\b(?:shows|obtains)\s+"([^"]+)"', stmt_clean)
    if shows_match:
        aspect_statement = shows_match.group(1).strip()
    else:
        colon_idx = stmt_clean.find(":")
        if colon_idx != -1:
            prop_part = stmt_clean[colon_idx + 1:].strip()
            # Handle shows after colon without quotes, or with quotes
            inner_shows = re.match(r'^(?:shows|obtains)\s+"([^"]+)"', prop_part)
            if inner_shows:
                aspect_statement = inner_shows.group(1).strip()
            elif prop_part.startswith('"') and prop_part.endswith('"'):
                aspect_statement = prop_part[1:-1].strip()
            elif prop_part.startswith('`') and prop_part.endswith('`'):
                aspect_statement = prop_part[1:-1].strip()
            else:
                aspect_statement = prop_part
        else:
            # Fallback: remove lemma or definition keyword and keep
            aspect_statement = re.sub(
                r'^(lemma|theorem|corollary|proposition|schematic_goal|definition|fun|primrec|function|datatype|type_synonym|inductive|coinductive|record|abbreviation)\s+',
                '', stmt_clean
            )
        
    # 2. Context aspect: theory imports + locale + entry abstract
    context_parts = []
    if theory_header:
        context_parts.append(f"Theory context: {theory_header.strip()}")
    if entry_metadata:
        if entry_metadata.get("title"):
            context_parts.append(f"AFP Entry: {entry_metadata['title']}")
        if entry_metadata.get("abstract"):
            context_parts.append(f"Abstract: {entry_metadata['abstract']}")
        if entry_metadata.get("topics"):
            context_parts.append(f"Topics: {', '.join(entry_metadata['topics'])}")
    aspect_context = "\n\n".join(context_parts) if context_parts else f"Theory: {lemma.get('theory', 'unknown')}"
    
    # 3. Strategy aspect: extract proof methods and classify style
    keyword = lemma.get("keyword", "")
    found_strategies = []
    
    # Label if this is a definition
    if keyword in {"definition", "fun", "primrec", "function", "datatype", "type_synonym", "inductive", "coinductive", "record", "abbreviation"}:
        found_strategies.append(f"{keyword}-style definition")
        
    methods = [
        "induction", "coinduction", "cases", "simp_all", "simp", "auto", "blast", "metis",
        "fastforce", "force", "clarify", "clarsimp", "safe", "rule", "subst", "linarith",
        "presburger", "ring", "algebra", "arith", "pat_completeness", "relation", "computation"
    ]
    proof_lower = proof.lower()
    for m in methods:
        if re.search(rf'\b{m}\b', proof_lower):
            found_strategies.append(m)
            
    if "proof" in proof_lower:
        found_strategies.append("structured proof")
    if "sorry" in proof_lower:
        found_strategies.append("sorry (unfinished)")
    if "oops" in proof_lower:
        found_strategies.append("oops (abandoned)")
        
    # Determine proof style (Teddy's apply-style, isa-style, hybrid, or simple)
    if proof.strip():
        has_apply = "apply" in proof_lower
        has_isar = any(kw in proof_lower for kw in ["proof", "qed", "show", "have", "fix", "assume", "obtain"])
        if has_apply and has_isar:
            style = "hybrid-style"
        elif has_apply:
            style = "apply-style"
        elif has_isar:
            style = "isar-style"
        else:
            style = "simple-style"
        found_strategies.append(style)
        
    aspect_strategy = ", ".join(found_strategies) if found_strategies else "unknown (direct or simple)"
    
    # 4. Dependencies aspect: extract cited theorems/lemmas
    deps = set()
    
    # simp add: / del: / only:
    for m in re.finditer(r'\bsimp\s+(?:add|del|only):\s*([a-zA-Z0-9_\'\.\s]+)', proof):
        for word in re.findall(r'\b[a-zA-Z0-9_\'\.]+\b', m.group(1)):
            if not word.isdigit() and word not in {"add", "del", "only", "simp"}:
                deps.add(word)
                
    # using X / unfolding X
    for m in re.finditer(r'\b(?:using|unfolding)\s+([a-zA-Z0-9_\'\.\s]+)', proof):
        for word in re.findall(r'\b[a-zA-Z0-9_\'\.]+\b', m.group(1)):
            if not word.isdigit() and word not in {"using", "unfolding"}:
                deps.add(word)
                
    # metis / blast / rule / subst / insert X
    for m in re.finditer(r'\b(?:metis|blast|rule|subst|insert)\s+([a-zA-Z0-9_\'\.\s]+)', proof):
        for word in re.findall(r'\b[a-zA-Z0-9_\'\.]+\b', m.group(1)):
            if not word.isdigit() and word not in {"metis", "blast", "rule", "subst", "insert", "no_types", "full_types"}:
                deps.add(word)
                
    ignored_deps = set(methods) | {
        "apply", "by", "proof", "qed", "sorry", "oops", "done", "defer", "prefer",
        "of", "where", "in", "and", "then", "hence", "thus", "show", "have", "obtain",
        "intro", "elim", "dest", "simp", "auto", "blast", "metis", "rule", "intro_classes",
        "standard", "this", "that", "unfolding", "using"
    }
    final_deps = {d for d in deps if d not in ignored_deps}
    aspect_dependencies = ", ".join(sorted(final_deps)) if final_deps else "none"
    
    return {
        "aspect_statement": aspect_statement,
        "aspect_context": aspect_context,
        "aspect_strategy": aspect_strategy,
        "aspect_dependencies": aspect_dependencies,
    }
