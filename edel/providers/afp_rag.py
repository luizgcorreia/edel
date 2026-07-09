"""Stage 1 data provider using pre-built AFP RAG index (lemmas/definitions level)."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from edel.providers.base import ensure_schema
from edel.il.index import NumpyRAGIndex

def generate_dataset(config: dict) -> tuple[pd.DataFrame, dict]:
    """Load AFP RAG index and return a lemma-level dataset.
    
    Expected config:
    {
        "provider": {
            "type": "afp_rag",
            "params": {
                "index_dir": "artifacts/rag_index"
            }
        }
    }
    """
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    index_dir = params.get("index_dir", "artifacts/rag_index")
    
    # 1. Load RAG Index
    index = NumpyRAGIndex()
    index.load(index_dir)
    
    if not index.metadata:
        raise ValueError(f"RAG Index at '{index_dir}' is empty or not loaded.")
        
    # Convert metadata to DataFrame
    df = pd.DataFrame(index.metadata)
    
    # Save original metadata fields to restore them after ensure_schema
    original_cols = {}
    for col in ["theory", "file", "line", "proof_text", "statement_text"]:
        if col in df.columns:
            original_cols[col] = df[col].copy()
            
    # Enrich title with theory name if not already qualified
    def enrich_title(row):
        title = row.get("title", "")
        theory = row.get("theory", "")
        if theory and theory not in title:
            return f"{theory}.{title}"
        return title

    df["title"] = df.apply(enrich_title, axis=1)
    
    # 2. Calculate "cited_by_count" (epistemic significance) by counting uses in AFP
    print("Calculating epistemic significance (citation counts) for lemmas/definitions...")
    citation_counts = resolve_and_count_citations(index.metadata)
    df["cited_by_count"] = df["title"].map(citation_counts).fillna(0).astype(int)
    
    # 3. Fill schema requirements
    # 'id' is required and should be unique. We use the qualified title of the lemma.
    df["id"] = df["title"]
    # 'abstract_text' is required. Let's use proof_text or statement_text
    df["abstract_text"] = df.get("proof_text", "")
    df["source_provider"] = "afp_rag"
    df["type"] = "lemma"
    df["theories"] = df.apply(lambda r: [r["theory"]] if isinstance(r.get("theory"), str) else [], axis=1)
    
    # Ensure it meets the required schema columns
    df = ensure_schema(df, provider_name="afp_rag")
    
    # Restore original metadata columns so they aren't lost by ensure_schema
    for col, series in original_cols.items():
        df[col] = series
    
    # 4. Add embeddings from the index to the DataFrame (after ensure_schema, to prevent them from being dropped)
    # Note: we serialize them to JSON strings to match normal Stage 3 outputs
    for aspect in ["problem", "method", "finding", "interpretation"]:
        emb_col = f"{aspect}_embedding"
        matrix = index.embeddings.get(aspect)
        if matrix is not None:
            # Convert each row vector to a JSON list string
            df[emb_col] = [json.dumps(row.tolist()) for row in matrix]
        else:
            df[emb_col] = None
    
    return df, {}

def resolve_and_count_citations(metadata: list[dict]) -> dict[str, int]:
    """Resolve references and count how many times each lemma/definition is used.
    
    Returns a dict mapping lemma title -> citation count.
    """
    from collections import defaultdict
    
    # 1. Map base names and qualified names to the full lemma titles.
    title_to_record = {}
    base_to_titles = defaultdict(list)
    qual2_to_titles = defaultdict(list)  # e.g., 'Theory.Lemma'
    
    for r in metadata:
        title = r["title"]
        title_to_record[title] = r
        parts = title.split(".")
        base = parts[-1]
        base_to_titles[base].append(title)
        if len(parts) >= 2:
            qual2 = ".".join(parts[-2:])
            qual2_to_titles[qual2].append(title)
            
    citation_counts = defaultdict(int)
    
    # 2. Process each lemma's dependencies to count references
    for r in metadata:
        theory = r.get("theory", "")
        session = theory.split(".")[0] if "." in theory else theory
        
        # Dependencies are in 'interpretation' (comma-separated list of tokens)
        deps_str = r.get("interpretation", "")
        if not deps_str or deps_str == "none" or deps_str.startswith("No specific reference theorem"):
            continue
            
        prefix = "This theorem represents or relies on the reference theorem: "
        if deps_str.startswith(prefix):
            deps_str = deps_str[len(prefix):]
            
        deps = [d.strip() for d in deps_str.split(",") if d.strip()]
        
        for dep in deps:
            resolved_title = None
            
            # Match 1: Is it a fully qualified title?
            if dep in title_to_record:
                resolved_title = dep
            # Match 2: Is it a 2-part qualified name (e.g. 'Theory.Lemma')?
            elif dep in qual2_to_titles:
                titles = qual2_to_titles[dep]
                if len(titles) == 1:
                    resolved_title = titles[0]
                else:
                    # Resolve to the one in the same session, if any
                    for t in titles:
                        if t.startswith(session + "."):
                            resolved_title = t
                            break
                    if not resolved_title:
                        resolved_title = titles[0]
            # Match 3: Is it a base name?
            elif dep in base_to_titles:
                titles = base_to_titles[dep]
                if len(titles) == 1:
                    resolved_title = titles[0]
                else:
                    # Resolve to same theory first
                    for t in titles:
                        if t.startswith(theory + "."):
                            resolved_title = t
                            break
                    # Resolve to same session next
                    if not resolved_title:
                        for t in titles:
                            if t.startswith(session + "."):
                                resolved_title = t
                                break
                    if not resolved_title:
                        resolved_title = titles[0]
                        
            if resolved_title:
                citation_counts[resolved_title] += 1
                
    return citation_counts
