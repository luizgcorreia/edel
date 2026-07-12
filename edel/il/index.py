"""Numpy-based vector index for Isabelle/AFP lemmas."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


class NumpyRAGIndex:
    """Numpy-based cosine similarity vector index.
    
    Manages both Lemma Space (with 4 aspects) and a dedicated Definition Space.
    Supports dynamic run-time additions for lemmas and definitions.
    """

    def __init__(self):
        # Lemma Space Static Data
        self.metadata: list[dict[str, Any]] = []
        self.embeddings: dict[str, np.ndarray | None] = {
            "problem": None,
            "method": None,
            "finding": None,
            "interpretation": None,
        }
        
        # Lemma Space Live Session Data
        self.live_metadata: list[dict[str, Any]] = []
        self.live_embeddings: dict[str, list[list[float]]] = {
            "problem": [],
            "method": [],
            "finding": [],
            "interpretation": [],
        }

    def save(self, directory: str | Path):
        """Save the index to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save lemma metadata and embeddings
        meta_df = pd.DataFrame(self.metadata)
        meta_df.to_parquet(directory / "metadata.parquet", index=False)
        
        npz_kwargs = {}
        for aspect, arr in self.embeddings.items():
            if arr is not None:
                npz_kwargs[aspect] = arr
        np.savez_compressed(directory / "embeddings.npz", **npz_kwargs)

        # Cleanup legacy separate definition files if they exist
        def_meta_path = directory / "definitions_metadata.parquet"
        def_emb_path = directory / "definitions_embeddings.npz"
        if def_meta_path.exists():
            try:
                def_meta_path.unlink()
            except Exception:
                pass
        if def_emb_path.exists():
            try:
                def_emb_path.unlink()
            except Exception:
                pass
            
        print(f"Saved RAG index to {directory} ({len(self.metadata)} items)")

    def load(self, directory: str | Path):
        """Load the index from a directory."""
        directory = Path(directory)
        if not (directory / "metadata.parquet").exists() or not (directory / "embeddings.npz").exists():
            raise FileNotFoundError(f"RAG index files not found in {directory}")
            
        # Load lemmas/definitions
        meta_df = pd.read_parquet(directory / "metadata.parquet")
        if "dependents_count" not in meta_df.columns:
            meta_df["dependents_count"] = 0
        self.metadata = meta_df.to_dict(orient="records")
        
        with np.load(directory / "embeddings.npz") as data:
            for aspect in self.embeddings.keys():
                if aspect in data:
                    self.embeddings[aspect] = data[aspect]

        # Clean legacy definition files if found during load
        def_meta_path = directory / "definitions_metadata.parquet"
        def_emb_path = directory / "definitions_embeddings.npz"
        if def_meta_path.exists():
            try:
                def_meta_path.unlink()
            except Exception:
                pass
        if def_emb_path.exists():
            try:
                def_emb_path.unlink()
            except Exception:
                pass
            
        print(f"Loaded RAG index from {directory} ({len(self.metadata)} items)")

    def build_from_dataframe(self, df: pd.DataFrame):
        """Build the index from an embedded DataFrame."""
        meta_cols = [c for c in df.columns if not c.endswith("_embedding")]
        self.metadata = df[meta_cols].to_dict(orient="records")
        
        aspects = ["problem", "method", "finding", "interpretation"]
        for aspect in aspects:
            emb_col = f"{aspect}_embedding"
            if emb_col in df.columns and not df.empty:
                embs = []
                for val in df[emb_col]:
                    if isinstance(val, str):
                        embs.append(json.loads(val))
                    else:
                        embs.append(val)
                self.embeddings[aspect] = np.array(embs, dtype=np.float32)
            else:
                self.embeddings[aspect] = None

    def update_dependents_counts(self):
        """Recalculate transitive dependents count (landscape height) for all lemmas and definitions in the unified index."""
        # 1. Build Whitelist & Name Maps
        all_items = self.metadata + self.live_metadata
        all_titles = set(item["title"] for item in all_items)
        
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
        
        for item in all_items:
            title = item["title"]
            keyword = item.get("keyword", "lemma")
            
            if keyword in DEF_KEYWORDS:
                # Process definition dependents (via dependents field)
                dependents_str = item.get("dependents", "none")
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
                cited_str = item.get("cited_deps", "none")
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
            
        # 4. Save counts back to metadata dicts in-place
        for item in all_items:
            item["dependents_count"] = dependents_counts.get(item["title"], 0)

    def add_live_lemma(
        self,
        name: str,
        aspect_text_dict: dict[str, str],
        embeddings_dict: dict[str, list[float]],
        theory: str,
        file: str = "",
        line: int = 0,
        proof_text: str = "",
        cited_deps: list[str] = [],
        keyword: str = "lemma",
        dependents: str = "none",
    ):
        """Add a newly proven lemma to the in-memory session index."""
        lemma_id = f"{theory}.{name}" if name else f"{theory}.live_lemma_{len(self.live_metadata)}"
        
        record = {
            "title": lemma_id,
            "problem":         aspect_text_dict.get("problem", ""),
            "method":          aspect_text_dict.get("method", ""),
            "finding":         aspect_text_dict.get("finding", ""),
            "interpretation":  aspect_text_dict.get("interpretation", ""),
            "theory":          theory,
            "keyword":         keyword,
            "file":            file,
            "line":            line,
            "proof_text":      proof_text,
            "statement_text":  aspect_text_dict.get("problem", ""),
            "cited_deps":      ", ".join(cited_deps) if cited_deps else "none",
            "dependents":      dependents,
            "source":          "live",
        }
        
        self.live_metadata.append(record)
        
        # Store embeddings
        for aspect in ["problem", "method", "finding", "interpretation"]:
            emb = embeddings_dict.get(aspect)
            if emb:
                self.live_embeddings[aspect].append(emb)
                
        # Recalculate dependents counts for all nodes in the unified index
        self.update_dependents_counts()

    def add_live_definition(
        self,
        name: str,
        statement_text: str,
        embedding: list[float],
        theory: str,
        file: str = "",
        line: int = 0,
        dependents: str = "none"
    ):
        """Add a newly defined construct to the in-memory session index as a 0-simplex."""
        aspect_text_dict = {
            "problem": statement_text,
            "method": statement_text,
            "finding": statement_text,
            "interpretation": statement_text,
        }
        embeddings_dict = {
            "problem": embedding,
            "method": embedding,
            "finding": embedding,
            "interpretation": embedding,
        }
        self.add_live_lemma(
            name=name,
            aspect_text_dict=aspect_text_dict,
            embeddings_dict=embeddings_dict,
            theory=theory,
            file=file,
            line=line,
            keyword="definition",
            dependents=dependents,
        )

    def persist_live_lemmas(self, directory: str | Path):
        """Merge live session items into the static index and save to disk."""
        if self.live_metadata:
            cleaned_live_meta = []
            for meta in self.live_metadata:
                cleaned = meta.copy()
                cleaned.pop("source", None)
                cleaned_live_meta.append(cleaned)
            self.metadata.extend(cleaned_live_meta)
            
            for aspect in ["problem", "method", "finding", "interpretation"]:
                live_list = self.live_embeddings[aspect]
                if not live_list:
                    continue
                live_arr = np.array(live_list, dtype=np.float32)
                
                static_arr = self.embeddings[aspect]
                if static_arr is None or len(static_arr) == 0:
                    self.embeddings[aspect] = live_arr
                else:
                    self.embeddings[aspect] = np.vstack([static_arr, live_arr])
                    
            self.live_metadata = []
            for aspect in self.live_embeddings:
                self.live_embeddings[aspect] = []
                
        self.save(directory)

    def search(
        self,
        query_vector: list[float],
        aspect: str = "problem",
        max_results: int = 10,
        theory_filter: str = "",
        sort_by_significance: bool = False,
        min_dependents: int = 0,
        exclude_definitions: bool = True,
        only_definitions: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the unified static and live indices by cosine similarity, optionally filtering definitions."""
        results = []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm
            
        DEF_KEYWORDS = {
            "definition", "fun", "primrec", "function", "datatype", "type_synonym",
            "inductive", "coinductive", "record", "abbreviation"
        }

        # 1. Search Static Index
        static_matrix = self.embeddings.get(aspect)
        if static_matrix is not None and len(static_matrix) > 0:
            norms = np.linalg.norm(static_matrix, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            norm_matrix = static_matrix / norms
            scores = norm_matrix @ q
            
            for idx, score in enumerate(scores):
                meta = self.metadata[idx]
                if theory_filter and theory_filter.lower() not in meta.get("theory", "").lower():
                    continue
                is_def = meta.get("keyword") in DEF_KEYWORDS
                if exclude_definitions and is_def:
                    continue
                if only_definitions and not is_def:
                    continue
                dep_count = int(meta.get("dependents_count", 0))
                if dep_count < min_dependents:
                    continue
                results.append({
                    "lemma": meta,
                    "score": float(score),
                    "source": "static",
                })
                
        # 2. Search Live Session Index
        live_list = self.live_embeddings.get(aspect, [])
        if live_list:
            live_matrix = np.array(live_list, dtype=np.float32)
            norms = np.linalg.norm(live_matrix, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            norm_matrix = live_matrix / norms
            scores = norm_matrix @ q
            
            for idx, score in enumerate(scores):
                meta = self.live_metadata[idx]
                if theory_filter and theory_filter.lower() not in meta.get("theory", "").lower():
                    continue
                is_def = meta.get("keyword") in DEF_KEYWORDS
                if exclude_definitions and is_def:
                    continue
                if only_definitions and not is_def:
                    continue
                dep_count = int(meta.get("dependents_count", 0))
                if dep_count < min_dependents:
                    continue
                results.append({
                    "lemma": meta,
                    "score": float(score),
                    "source": "live",
                })
                
        # Apply Epistemic Bias Re-ranking
        if sort_by_significance and results:
            import math
            log_deps = []
            for r in results:
                meta = r["lemma"]
                dep_count = int(meta.get("dependents_count", 0))
                log_deps.append(math.log1p(dep_count))
            max_log_dep = max(log_deps) if log_deps else 0.0
            if max_log_dep > 1e-10:
                for r, log_dep in zip(results, log_deps):
                    r["score"] = r["score"] + 0.15 * (log_dep / max_log_dep)
                    
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def search_definitions(
        self,
        query_vector: list[float],
        max_results: int = 10,
        theory_filter: str = "",
        sort_by_significance: bool = False,
        min_dependents: int = 0,
    ) -> list[dict[str, Any]]:
        """Search the Definition Space (wrapped query over unified index)."""
        raw_hits = self.search(
            query_vector=query_vector,
            aspect="problem",
            max_results=max_results,
            theory_filter=theory_filter,
            sort_by_significance=sort_by_significance,
            min_dependents=min_dependents,
            exclude_definitions=False,
            only_definitions=True,
        )
        # Map output to expect "definition" key for server formatting/compatibility
        results = []
        for h in raw_hits:
            results.append({
                "definition": h["lemma"],
                "score": h["score"],
                "source": h["source"],
            })
        return results
