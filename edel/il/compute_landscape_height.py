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
    definitions_path = index_dir / "definitions_metadata.parquet"
    
    if not metadata_path.exists():
        print(f"[Landscape Height] Error: Lemma metadata not found at {metadata_path}")
        return
        
    print(f"[Landscape Height] Loading metadata from {index_dir}...")
    df_lemmas = pd.read_parquet(metadata_path)
    
    df_defs = None
    if definitions_path.exists():
        df_defs = pd.read_parquet(definitions_path)
        
    # 1. Build Whitelist & Name Maps
    lemma_titles = set(df_lemmas["title"].tolist())
    def_titles = set(df_defs["title"].tolist()) if df_defs is not None else set()
    all_titles = lemma_titles | def_titles
    
    # Map short names (e.g. "append_assoc") to list of fully qualified titles (e.g. ["HOL.List.append_assoc"])
    short_name_map: dict[str, list[str]] = {}
    for title in all_titles:
        short = title.split(".")[-1]
        if short:
            short_name_map.setdefault(short, []).append(title)
            
    # 2. Build Dependency Graph G (node -> set of dependencies)
    # And Transpose Graph G^T (dependency -> set of direct dependents)
    # Nodes in graph will be all titles
    adj_transpose: dict[str, set[str]] = {title: set() for title in all_titles}
    
    # A. Process Lemmas (via cited_deps)
    for _, row in df_lemmas.iterrows():
        title = row["title"]
        cited_str = row.get("cited_deps", "none")
        if not cited_str or cited_str == "none":
            continue
            
        cites = [c.strip() for c in cited_str.split(",") if c.strip()]
        for cite in cites:
            # Check exact match
            if cite in all_titles:
                adj_transpose[cite].add(title)
            # Check short name match
            elif cite in short_name_map:
                for target_title in short_name_map[cite]:
                    adj_transpose[target_title].add(title)
                    
    # B. Process Definitions (via dependents field)
    if df_defs is not None:
        for _, row in df_defs.iterrows():
            def_title = row["title"]
            dependents_str = row.get("dependents", "none")
            if not dependents_str or dependents_str == "none":
                continue
                
            deps = [d.strip() for d in dependents_str.split(",") if d.strip()]
            for dep in deps:
                if dep in all_titles:
                    adj_transpose[def_title].add(dep)
                elif dep in short_name_map:
                    for target_title in short_name_map[dep]:
                        adj_transpose[def_title].add(target_title)

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
    
    # 4. Save counts back to dataframes
    df_lemmas["dependents_count"] = df_lemmas["title"].map(dependents_counts).fillna(0).astype(int)
    df_lemmas.to_parquet(metadata_path, index=False)
    print(f"[Landscape Height] Updated lemma metadata with dependents_count.")
    
    if df_defs is not None:
        df_defs["dependents_count"] = df_defs["title"].map(dependents_counts).fillna(0).astype(int)
        df_defs.to_parquet(definitions_path, index=False)
        print(f"[Landscape Height] Updated definitions metadata with dependents_count.")


def main():
    parser = argparse.ArgumentParser(description="Compute landscape height (dependents count) for RAG index.")
    parser.add_argument("--index-dir", default="artifacts/rag_index", help="RAG index directory")
    args = parser.parse_args()
    
    compute_and_save_landscape_height(args.index_dir)


if __name__ == "__main__":
    main()
