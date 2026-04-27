"""AFP data provider using the Archive of Formal Proofs repository."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import re
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

STRICT_STOPWORDS = {
    "get", "set", "add", "is", "of", "to", "in", "on", "for", "as", "by", "new",
    "init", "start", "top", "bot", "empty", "proj", "update", "clear", "local",
    "less", "makes", "other", "eq", "do", "val", "fun", "let", "where"
}


def strip_html(text: str | None) -> str:
    """Remove HTML tags from a string."""
    if not text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def get_vowel_ratio(token: str) -> float:
    """Calculate the ratio of vowels in a string."""
    token = token.lower()
    vowels = sum(1 for c in token if c in "aeiou")
    return vowels / len(token) if token else 0.0


def is_meaningful_semantic_token(
    nt: str, meaningful_acronyms: set[str] | None = None
) -> bool:
    """Apply 4-layer filtering logic to a normalized token."""
    # Layer 1: Total Length (strict for embeddings)
    if len(nt) < 4:
        return False

    # Layer 2: Domain-specific Stopwords
    if nt in STRICT_STOPWORDS:
        return False

    words = nt.split()

    # Layer 3: Multi-word tokens are usually phrases (e.g., "hash tree")
    if len(words) > 1:
        # Check if it's not just a soup of single letters like "f g r"
        if all(len(w) < 3 for w in words):
            return False
        return True

    # Layer 4: Single word tokens must be "vowel-rich", "long", or "frequent"
    w = words[0]
    
    # Bypass for longer English/Technical words that might have low vowel density
    if len(w) >= 8:
        return True

    # Threshold 0.3 catches words with 1 vowel in 4 chars (0.25) like "extg" or "math"
    # These will require promotion via frequency analysis
    if get_vowel_ratio(w) < 0.3:
        if meaningful_acronyms and w in meaningful_acronyms:
            return True
        return False

    # Generic length check for single words to avoid code noise
    if len(w) < 4:
        return False

    return True


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


def normalize_token_list(
    tokens: list[str], meaningful_acronyms: set[str] | None = None
) -> list[str]:
    """Normalize and filter a list of tokens for embedding usage."""
    if not tokens:
        return []
    normalized = []
    seen = set()
    for t in tokens:
        nt = normalize_token(t)
        if nt and nt not in seen:
            if is_meaningful_semantic_token(nt, meaningful_acronyms):
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
    sampling_strategy = params.get("sampling_strategy", "probabilistic")
    sampling_seed = params.get("sampling_seed")

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
    
    # Pre-parse sampling for probabilistic to save time
    if limit and sampling_strategy == "probabilistic":
        import random
        rng = random.Random(sampling_seed)
        sorted_entry_ids = rng.sample(sorted_entry_ids, min(limit, len(sorted_entry_ids)))

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
            "id": entry_id,
            "title": meta["title"],
            "abstract_text": strip_html(meta["abstract"]),
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

    # 3. Compute Citations & Dynamic Acronym Promotion
    cited_by = defaultdict(int)
    for dep, dependents in dependency_graph.items():
        cited_by[dep] = len(dependents)

    # Pass 1: Global Frequency Analysis for Acronym Promotion
    # We identify "consonant soup" tokens that appear across multiple entries
    token_entry_map = defaultdict(set)
    for entry_id, data in raw_entries.items():
        all_raw_candidates = (
            data["definitions"] + data["lemmas"] + data["imports"] + data["theories"]
        )
        for t in all_raw_candidates:
            nt = normalize_token(t)
            if not nt:
                continue
            words = nt.split()
            for w in words:
                # Potential acronym if it has low vowel density
                if len(w) >= 3 and get_vowel_ratio(w) < 0.25:
                    token_entry_map[w].add(entry_id)

    # Meaningful acronyms appear in at least 3 unique entries
    meaningful_acronyms = {
        t for t, entries in token_entry_map.items() if len(entries) >= 3
    }

    # Post-parse sampling for deterministic (top-cited)
    if limit and sampling_strategy == "deterministic":
        # Sort by citation count descending, then alphabetically to break ties
        sorted_by_citations = sorted(
            raw_entries.keys(),
            key=lambda x: (cited_by.get(x, 0), x),
            reverse=True
        )
        top_entries = set(sorted_by_citations[:limit])
        raw_entries = {k: v for k, v in raw_entries.items() if k in top_entries}

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

        norm_defs = normalize_token_list(defs, meaningful_acronyms)
        norm_lems = normalize_token_list(lems, meaningful_acronyms)
        norm_imps = normalize_token_list(imps, meaningful_acronyms)
        norm_thys = normalize_token_list(thys, meaningful_acronyms)
        
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
        # We now KEEP the semantic entities for traceability as requested
        record = {k: v for k, v in data.items()}
        record["source_provider"] = afp_root.name
        record["primary_location"] = "Archive of Formal Proofs"
        record["type"] = "theory"
        record["language"] = "en"
        record["has_fulltext"] = True
        
        records.append(record)

    df = pd.DataFrame(records)
    return ensure_schema(df, provider_name="afp")
