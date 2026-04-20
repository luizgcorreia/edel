"""AFP data provider using the Archive of Formal Proofs repository."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from edel.io.afp import (
    ensure_afp_repo,
    load_afp_authors,
    load_afp_metadata,
    parse_afp_root,
    parse_thy_entities,
)
from edel.providers.base import (
    ensure_schema,
    normalize_token,
    stratified_sample,
    unique_preserve_order,
)

BAD_CONTEXT_TOKENS = {
    "document",
    "false",
    "theories",
    "example",
    "examples",
    "test",
    "tests",
    "misc",
    "setup",
    "core",
    "basis",
    "library",
    "lib",
    "common",
    "preliminaries",
    "main",
}


def clean_imports(imports: list[str]) -> list[str]:
    """Clean and filter import tokens."""
    imports = unique_preserve_order(imports)
    cleaned = []
    for x in imports:
        x_clean = x.lower()
        if "/" in x or "." in x:
            continue
        if x_clean in BAD_CONTEXT_TOKENS:
            continue
        cleaned.append(x)
    return cleaned


def clean_lemmas(lemmas: list[str]) -> list[str]:
    """Clean and filter lemma/theorem tokens."""
    import re

    BAD_LEMMA_TOKENS = {"assumes", "shows", "fixes"}
    lemmas = unique_preserve_order(lemmas)
    cleaned = []
    for l in lemmas:
        l_clean = l.lower()
        if len(l_clean) < 4:
            continue
        if l_clean in BAD_LEMMA_TOKENS:
            continue
        if l_clean.startswith(
            ("aux", "tmp", "helper", "self", "example", "lem", "lemma", "theorem")
        ):
            continue
        if re.match(r"(lem|lemma|theorem)\d+", l_clean):
            continue
        cleaned.append(l)
    return cleaned


def clean_definitions(defs: list[str]) -> list[str]:
    """Clean and filter definition tokens."""
    defs = unique_preserve_order(defs)
    return [d for d in defs if len(d) > 2]


def clean_theories(theories: list[str]) -> list[str]:
    """Clean and filter theory file tokens."""
    theories = unique_preserve_order(theories)
    cleaned = []
    for t in theories:
        if not isinstance(t, str):
            continue
        t_clean = t.lower()
        if "." in t:
            continue
        if t_clean in BAD_CONTEXT_TOKENS:
            continue
        cleaned.append(t)
    return cleaned


def normalize_token_list(tokens: list[str]) -> list[str]:
    """Normalize a list of tokens for embedding usage."""
    if not tokens:
        return []
    normalized = []
    seen = set()
    for t in tokens:
        nt = normalize_token(t)
        if nt and len(nt) > 2 and nt not in seen:
            normalized.append(nt)
            seen.add(nt)
    return normalized


def build_segment_text(label: str, tokens: list[str]) -> str:
    """Format tokens into a labeled text segment."""
    if not tokens:
        return ""
    return f"{label}:\n" + "\n".join(tokens)


def generate_dataset(config: dict) -> pd.DataFrame:
    """Harvest AFP entries and extract semantic data.
    
    Expected config:
    {
        "provider": {
            "type": "afp",
            "repo_url": "https://foss.heptapod.net/isa-afp/afp-2025-2",
            "params": {
                "n_documents": int (optional)
            }
        }
    }
    """
    provider_cfg = config.get("provider", {})
    repo_url = provider_cfg.get("repo_url")
    params = provider_cfg.get("params", {})
    limit = params.get("n_documents")

    if not repo_url:
        raise ValueError("repo_url must be specified in provider config for AFP.")

    # 1. Ensure Repo & Load Metadata
    afp_root = ensure_afp_repo(repo_url)
    author_map = load_afp_authors(afp_root)
    entries_metadata = load_afp_metadata(afp_root, author_map)

    # 2. Phase 1: Collect Entry Information & Dependencies
    thys_path = afp_root / "thys"
    entry_ids = set(entries_metadata.keys())
    
    dependency_graph = defaultdict(set)
    raw_entries = {}
    
    # We iterate over metadata entries to ensure we only process "official" entries
    sorted_entry_ids = sorted(list(entry_ids))
    if limit:
        sorted_entry_ids = sorted_entry_ids[:limit]

    print(f"Parsing {len(sorted_entry_ids)} AFP entries...")
    
    for entry_id in tqdm(sorted_entry_ids, desc="Parsing AFP"):
        entry_dir = thys_path / entry_id
        if not entry_dir.exists():
            continue

        # Parse ROOT file
        root_file = entry_dir / "ROOT"
        _, imports, theories = parse_afp_root(root_file)
        
        # Parse Theories
        definitions, lemmas = parse_thy_entities(entry_dir)
        
        # Metadata from TOML
        meta = entries_metadata[entry_id]
        
        # Store for citation counts and final dataset
        entry_data = {
            "id": f"afp:{entry_id}",
            "title": meta["title"],
            "abstract_text": meta["abstract"],
            "authorships": meta["authorships"],
            "publication_year": int(meta["date"].split("-")[0]) if "-" in meta["date"] else None,
            "topics": meta["topics"],
            "imports": clean_imports(imports),
            "theories": clean_theories(theories),
            "definitions": clean_definitions(definitions),
            "lemmas": clean_lemmas(lemmas),
        }
        
        raw_entries[entry_id] = entry_data
        
        # Build dependency graph
        for dep in entry_data["imports"]:
            if dep in entry_ids:
                dependency_graph[dep].add(entry_id)

    # 3. Compute Citations
    cited_by = defaultdict(int)
    for dep, dependents in dependency_graph.items():
        cited_by[dep] = len(dependents)

    # 4. Build Dataset & Epistemic Aspects
    records = []
    for entry_id, data in raw_entries.items():
        # Citation count
        data["cited_by_count"] = cited_by.get(entry_id, 0)
        
        # Prefill Epistemic Aspects (matches colab logic)
        defs = stratified_sample(data["definitions"], 15)
        lems = stratified_sample(data["lemmas"], 20)
        imps = data["imports"]
        thys = data["theories"]
        
        norm_defs = normalize_token_list(defs)
        norm_lems = normalize_token_list(lems)
        norm_imps = normalize_token_list(imps)
        norm_thys = normalize_token_list(thys)
        
        # Map to aspects
        data["problem"] = "" # AFP usually doesn't have a specific problem token set
        data["method"] = build_segment_text("definitions", norm_defs)
        data["finding"] = build_segment_text("key lemmas", norm_lems)
        data["interpretation"] = (
            build_segment_text("imports", norm_imps) + 
            "\n" + 
            build_segment_text("theories", norm_thys)
        ).strip()
        
        # Clean up internal parsing fields before adding to record
        record = {k: v for k, v in data.items() if k not in ["imports", "theories", "definitions", "lemmas"]}
        record["source_provider"] = "afp"
        record["primary_location"] = "Archive of Formal Proofs"
        record["type"] = "theory"
        record["language"] = "en"
        record["has_fulltext"] = True
        
        records.append(record)

    df = pd.DataFrame(records)
    return ensure_schema(df, provider_name="afp")
