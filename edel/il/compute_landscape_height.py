"""Script to compute the transitive dependents count (landscape height) for indexed lemmas and definitions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
from typing import Any


def compute_and_save_landscape_height(index_dir: str | Path) -> None:
    """Load index metadata, compute transitive dependents count, and save updated metadata."""
    index_dir = Path(index_dir)
    
    metadata_path = index_dir / "metadata.parquet"
    if not metadata_path.exists():
        print(f"[Landscape Height] Error: Metadata not found at {metadata_path}")
        return
        
    print(f"[Landscape Height] Loading metadata from {index_dir}...")
    df = pd.read_parquet(metadata_path)
    
    # 1. Build Whitelist & Name Maps
    all_titles = set(df["title"].tolist())
    
    # Map short names (e.g. "append_assoc") to list of fully qualified titles (e.g. ["HOL.List.append_assoc"])
    short_name_map: dict[str, list[str]] = {}
    for title in all_titles:
        short = title.split(".")[-1]
        if short:
            short_name_map.setdefault(short, []).append(title)
            
    # 2. Build Transpose Graph G^T (dependency -> set of direct dependents)
    adj_transpose: dict[str, set[str]] = {title: set() for title in all_titles}
    
    DEF_KEYWORDS = {
        "definition", "fun", "primrec", "function", "datatype", "type_synonym",
        "inductive", "coinductive", "record", "abbreviation"
    }
    
    for _, row in df.iterrows():
        title = row["title"]
        keyword = row.get("keyword", "lemma")
        
        if keyword in DEF_KEYWORDS:
            # Process definition dependents (via dependents field)
            dependents_str = row.get("dependents", "none")
            if dependents_str and dependents_str != "none":
                deps = [d.strip() for d in dependents_str.split(",") if d.strip()]
                for dep in deps:
                    if dep in all_titles:
                        adj_transpose[title].add(dep)
                    elif dep in short_name_map:
                        for target_title in short_name_map[dep]:
                            adj_transpose[title].add(target_title)
        else:
            # Process lemma dependencies (via cited_deps)
            cited_str = row.get("cited_deps", "none")
            if cited_str and cited_str != "none":
                cites = [c.strip() for c in cited_str.split(",") if c.strip()]
                for cite in cites:
                    if cite in all_titles:
                        adj_transpose[cite].add(title)
                    elif cite in short_name_map:
                        for target_title in short_name_map[cite]:
                            adj_transpose[target_title].add(title)

    # 3. Compute Transitive Dependents Count using Cycle-Safe DFS
    memo: dict[str, set[str]] = {}
    
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
        
    print(f"[Landscape Height] Computed transitive dependents for {len(all_titles)} nodes.")
    
    # 4. Save counts back to dataframe
    df["dependents_count"] = df["title"].map(dependents_counts).fillna(0).astype(int)
    df.to_parquet(metadata_path, index=False)
    print(f"[Landscape Height] Updated unified metadata with dependents_count.")


def main():
    parser = argparse.ArgumentParser(description="Compute landscape height (dependents count) for RAG index.")
    parser.add_argument("--index-dir", default="artifacts/rag_index", help="RAG index directory")
    args = parser.parse_args()
    
    compute_and_save_landscape_height(args.index_dir)


if __name__ == "__main__":
    main()
