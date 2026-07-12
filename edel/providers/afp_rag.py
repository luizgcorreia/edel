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
    
    # 2. Calculate "cited_by_count" (epistemic significance) by counting transitive dependents
    print("Calculating epistemic significance (transitive dependents count) for lemmas/definitions...")
    if "dependents_count" in df.columns and (df["dependents_count"] > 0).any():
        df["cited_by_count"] = df["dependents_count"].fillna(0).astype(int)
    elif "cited_deps" in df.columns:
        # Calculate transitive dependents count on the fly using the new format
        citation_counts = resolve_and_count_citations_transitive(
            index.metadata
        )
        df["cited_by_count"] = df["title"].map(citation_counts).fillna(0).astype(int)
    else:
        df["cited_by_count"] = 0
    
    # 3. Fill schema requirements
    # 'id' is required and should be unique. We use the qualified title of the lemma.
    df["id"] = df["title"]
    # 'abstract_text' is required. Concatenate the four aspects (separated by newlines) to match structured abstract task
    def build_abstract(row):
        parts = []
        for aspect in ["problem", "method", "finding", "interpretation"]:
            val = row.get(aspect, "")
            if val and str(val).lower() not in ("none", "unknown"):
                parts.append(str(val))
        if not parts:
            fallback = row.get("statement_text") or row.get("proof_text") or ""
            return str(fallback)
        return "\n".join(parts)
        
    df["abstract_text"] = df.apply(build_abstract, axis=1)
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

def resolve_and_count_citations_transitive(
    metadata: list[dict]
) -> dict[str, int]:
    """Compute the transitive dependents count (landscape height) for each lemma/definition.
    
    Uses cited_deps and dependents to construct the dependency graph, then runs a
    cycle-safe depth-first search to find the size of the transitive reachability set.
    """
    all_titles = set(r["title"] for r in metadata)
    
    DEF_KEYWORDS = {
        "definition", "fun", "primrec", "function", "datatype", "type_synonym",
        "inductive", "coinductive", "record", "abbreviation"
    }

    # Map short names (e.g. "append_assoc") to list of fully qualified titles
    short_name_map = {}
    for title in all_titles:
        short = title.split(".")[-1]
        if short:
            short_name_map.setdefault(short, []).append(title)
            
    # Build Transpose Graph G^T (dependency -> set of direct dependents)
    adj_transpose = {title: set() for title in all_titles}
    
    for r in metadata:
        title = r["title"]
        keyword = r.get("keyword", "lemma")
        
        if keyword in DEF_KEYWORDS:
            # Process definition dependents (via dependents field)
            dependents_str = r.get("dependents", "none")
            if dependents_str and dependents_str != "none":
                deps = [d.strip() for d in dependents_str.split(",") if d.strip()]
                for dep in deps:
                    if dep in adj_transpose:
                        adj_transpose[title].add(dep)
                    elif dep in short_name_map:
                        for target_title in short_name_map[dep]:
                            adj_transpose[title].add(target_title)
        else:
            # Process Lemma (via cited_deps)
            cited_str = r.get("cited_deps", "none")
            if cited_str and cited_str != "none":
                cites = [c.strip() for c in cited_str.split(",") if c.strip()]
                for cite in cites:
                    # Check exact match
                    if cite in adj_transpose:
                        adj_transpose[cite].add(title)
                    # Check short name match
                    elif cite in short_name_map:
                        for target_title in short_name_map[cite]:
                            adj_transpose[target_title].add(title)
                            
    # Compute Transitive Dependents Count using Cycle-Safe DFS
    memo = {}
    
    def dfs(node: str, path: set[str]) -> set[str]:
        if node in memo:
            return memo[node]
        if node in path:
            return set()  # Cycle detected
            
        path.add(node)
        reachable = {node}
        for neighbor in adj_transpose.get(node, []):
            reachable.update(dfs(neighbor, path))
        path.remove(node)
        
        memo[node] = reachable
        return reachable
        
    dependents_counts = {}
    for node in all_titles:
        dependents_counts[node] = len(dfs(node, set())) - 1  # Exclude self
        
    return dependents_counts

