"""Isabelle segment parser and grouping logic."""

from __future__ import annotations

import re
from typing import Any


def parse_source_segments(raw_source: str) -> dict[int, str]:
    """Parse raw Ir.source output (which may contain YXML markup) into segment texts.
    
    Handles multi-line continuation of segments.
    """
    segments = {}
    current_idx = None
    current_lines = []
    
    # We strip control characters but keep raw newlines/spaces
    def clean_yxml(t):
        return t.replace("\x05", "").replace("\x06", "")
        
    for line in raw_source.splitlines():
        plain = clean_yxml(line).lstrip()
        # Segment starts with index prefix, e.g. "   6  lemma ..."
        idx_match = re.match(r'^(\d+)\s', plain)
        if idx_match:
            # Save previous segment
            if current_idx is not None:
                segments[current_idx] = "\n".join(current_lines).strip()
            current_idx = int(idx_match.group(1))
            # Strip the index prefix from display line
            content = re.sub(r'^\s*\d+\s{1,2}', '', clean_yxml(line).lstrip())
            current_lines = [content]
        else:
            if current_idx is not None:
                current_lines.append(line)
                
    if current_idx is not None:
        segments[current_idx] = "\n".join(current_lines).strip()
        
    return segments


def extract_lemma_name(stmt: str) -> str:
    """Extract the formal name of a lemma from its declaration statement."""
    stmt = stmt.strip()
    stmt_clean = re.sub(r'\s+', ' ', stmt)
    # Matches: lemma name: or lemma name [simp, intro]:
    m = re.match(r'^(?:lemma|theorem|corollary|proposition|schematic_goal)\s+([a-zA-Z0-9_\'\.]+)(?:\s+\[[^\]]*\])?\s*:', stmt_clean)
    if m:
        return m.group(1)
    return ""


def extract_definition_name(stmt: str, keyword: str) -> str:
    """Extract the formal name of a definition/function from its declaration statement."""
    stmt = stmt.strip()
    stmt_clean = re.sub(r'\s+', ' ', stmt)
    m = re.match(rf'^(?:{keyword})\s+([a-zA-Z0-9_\'\.]+)', stmt_clean)
    if m:
        return m.group(1)
    return ""


def group_segments_to_lemmas(seg_map: dict[int, dict], segments: dict[int, str]) -> list[dict[str, Any]]:
    """Group sequential segments into logical lemma and definition units with statement and proof.
    
    Args:
        seg_map: Dict of {idx: {keyword, line, offset, file}}
        segments: Dict of {idx: segment_text}
    """
    units = []
    current_unit = None
    
    indices = sorted(segments.keys())
    
    LEMMA_KEYWORDS = {"lemma", "theorem", "corollary", "proposition", "schematic_goal"}
    DEF_KEYWORDS = {
        "definition", "fun", "primrec", "function", "datatype", "type_synonym",
        "inductive", "coinductive", "record", "abbreviation"
    }
    ALL_INGEST_KEYWORDS = LEMMA_KEYWORDS | DEF_KEYWORDS
    
    DECL_KEYWORDS = {
        "lemma", "theorem", "corollary", "proposition", "schematic_goal",
        "definition", "fun", "primrec", "function", "datatype", "type_synonym",
        "inductive", "coinductive", "record", "abbreviation", "class", "instantiation",
        "locale", "context", "end", "theory"
    }
    PROOF_KEYWORDS = {"by", "apply", "proof", "qed", "sorry", "oops", "done", "defer", "prefer", "using", "unfolding"}
    
    for idx in indices:
        seg_info = seg_map.get(idx, {})
        keyword = seg_info.get("keyword", "")
        text = segments[idx]
        
        if keyword in ALL_INGEST_KEYWORDS:
            if current_unit:
                units.append(current_unit)
            name = extract_lemma_name(text) if keyword in LEMMA_KEYWORDS else extract_definition_name(text, keyword)
            current_unit = {
                "name": name,
                "keyword": keyword,
                "theory": seg_info.get("theory", ""),
                "file": seg_info.get("file", ""),
                "start_line": seg_info.get("line"),
                "segment_start": idx,
                "segment_end": idx,
                "statement_text": text,
                "proof_segments": [],
            }
        elif current_unit:
            # If we see another top-level declaration keyword, terminate current unit
            if keyword in DECL_KEYWORDS and keyword not in PROOF_KEYWORDS:
                units.append(current_unit)
                current_unit = None
            else:
                current_unit["proof_segments"].append(text)
                current_unit["segment_end"] = idx
                
                # If this segment terminates the proof
                if keyword in {"by", "qed", "sorry", "oops"}:
                    units.append(current_unit)
                    current_unit = None
                    
    if current_unit:
        units.append(current_unit)
        
    final_units = []
    for u in units:
        proof_text = "\n".join(u["proof_segments"]).strip()
        name_placeholder = f"{u['keyword']}_{u['segment_start']}"
        final_units.append({
            "id": f"{u['theory']}.{u['name']}" if u['name'] else f"{u['theory']}.{name_placeholder}",
            "name": u["name"],
            "keyword": u["keyword"],
            "theory": u["theory"],
            "file": u["file"],
            "line": u["start_line"],
            "segment_start": u["segment_start"],
            "segment_end": u["segment_end"],
            "statement_text": u["statement_text"],
            "proof_text": proof_text,
        })
    return final_units

