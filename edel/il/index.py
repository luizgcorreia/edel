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

        # Definition Space Static Data
        self.definition_metadata: list[dict[str, Any]] = []
        self.definition_embeddings: np.ndarray | None = None

        # Definition Space Live Session Data
        self.live_definition_metadata: list[dict[str, Any]] = []
        self.live_definition_embeddings: list[list[float]] = []

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

        # Save definition metadata and embeddings
        if self.definition_metadata:
            def_df = pd.DataFrame(self.definition_metadata)
            def_df.to_parquet(directory / "definitions_metadata.parquet", index=False)
        if self.definition_embeddings is not None:
            np.savez_compressed(directory / "definitions_embeddings.npz", definitions=self.definition_embeddings)
            
        print(f"Saved RAG index to {directory} ({len(self.metadata)} lemmas, {len(self.definition_metadata)} definitions)")

    def load(self, directory: str | Path):
        """Load the index from a directory."""
        directory = Path(directory)
        if not (directory / "metadata.parquet").exists() or not (directory / "embeddings.npz").exists():
            raise FileNotFoundError(f"RAG index files not found in {directory}")
            
        # Load lemmas
        meta_df = pd.read_parquet(directory / "metadata.parquet")
        if "dependents_count" not in meta_df.columns:
            meta_df["dependents_count"] = 0
        self.metadata = meta_df.to_dict(orient="records")
        
        with np.load(directory / "embeddings.npz") as data:
            for aspect in self.embeddings.keys():
                if aspect in data:
                    self.embeddings[aspect] = data[aspect]

        # Load definitions if they exist
        def_meta_path = directory / "definitions_metadata.parquet"
        if def_meta_path.exists():
            def_df = pd.read_parquet(def_meta_path)
            if "dependents_count" not in def_df.columns:
                def_df["dependents_count"] = 0
            self.definition_metadata = def_df.to_dict(orient="records")
        else:
            self.definition_metadata = []
            
        def_emb_path = directory / "definitions_embeddings.npz"
        if def_emb_path.exists():
            with np.load(def_emb_path) as data:
                if "definitions" in data:
                    self.definition_embeddings = data["definitions"]
        else:
            self.definition_embeddings = None
            
        print(f"Loaded RAG index from {directory} ({len(self.metadata)} lemmas, {len(self.definition_metadata)} definitions)")

    def build_from_dataframe(self, df: pd.DataFrame):
        """Build the index from an embedded DataFrame, partitioning lemmas and definitions."""
        DEF_KEYWORDS = {
            "definition", "fun", "primrec", "function", "datatype", "type_synonym",
            "inductive", "coinductive", "record", "abbreviation"
        }
        
        # Check and partition
        meta_cols = [c for c in df.columns if not c.endswith("_embedding")]
        
        if "keyword" in df.columns:
            def_mask = df["keyword"].isin(DEF_KEYWORDS)
            def_df = df[def_mask]
            lemma_df = df[~def_mask]
        else:
            def_df = df.iloc[0:0]
            lemma_df = df

        # 1. Build Definition Space
        self.definition_metadata = def_df[meta_cols].to_dict(orient="records")
        if not def_df.empty:
            embs = []
            for val in def_df["problem_embedding"]:
                if isinstance(val, str):
                    embs.append(json.loads(val))
                else:
                    embs.append(val)
            self.definition_embeddings = np.array(embs, dtype=np.float32)
        else:
            self.definition_embeddings = None

        # 2. Build Lemma Space
        self.metadata = lemma_df[meta_cols].to_dict(orient="records")
        aspects = ["problem", "method", "finding", "interpretation"]
        for aspect in aspects:
            emb_col = f"{aspect}_embedding"
            if emb_col in lemma_df.columns and not lemma_df.empty:
                embs = []
                for val in lemma_df[emb_col]:
                    if isinstance(val, str):
                        embs.append(json.loads(val))
                    else:
                        embs.append(val)
                self.embeddings[aspect] = np.array(embs, dtype=np.float32)
            else:
                self.embeddings[aspect] = None

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
            "keyword":         "lemma",
            "file":            file,
            "line":            line,
            "proof_text":      proof_text,
            "statement_text":  aspect_text_dict.get("problem", ""),
            "cited_deps":      ", ".join(cited_deps),
            "dependents":      "none",
            "source":          "live",
        }
        
        self.live_metadata.append(record)
        
        # Store embeddings
        for aspect in ["problem", "method", "finding", "interpretation"]:
            emb = embeddings_dict.get(aspect)
            if emb:
                self.live_embeddings[aspect].append(emb)

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
        """Add a newly defined construct to the in-memory session definition index."""
        def_id = f"{theory}.{name}" if name else f"{theory}.live_def_{len(self.live_definition_metadata)}"
        
        record = {
            "title":           def_id,
            "problem":         statement_text,
            "method":          "",
            "finding":         "",
            "interpretation":  "",
            "theory":          theory,
            "keyword":         "definition",
            "file":            file,
            "line":            line,
            "proof_text":      "",
            "statement_text":  statement_text,
            "cited_deps":      "none",
            "dependents":      dependents,
            "source":          "live",
        }
        
        self.live_definition_metadata.append(record)
        self.live_definition_embeddings.append(embedding)

    def persist_live_lemmas(self, directory: str | Path):
        """Merge live session lemmas and definitions into the static index and save to disk."""
        # 1. Merge Lemmas
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

        # 2. Merge Definitions
        if self.live_definition_metadata:
            cleaned_live_def_meta = []
            for meta in self.live_definition_metadata:
                cleaned = meta.copy()
                cleaned.pop("source", None)
                cleaned_live_def_meta.append(cleaned)
            self.definition_metadata.extend(cleaned_live_def_meta)
            
            live_arr = np.array(self.live_definition_embeddings, dtype=np.float32)
            static_arr = self.definition_embeddings
            if static_arr is None or len(static_arr) == 0:
                self.definition_embeddings = live_arr
            else:
                self.definition_embeddings = np.vstack([static_arr, live_arr])
                
            self.live_definition_metadata = []
            self.live_definition_embeddings = []
            
        self.save(directory)

    def search(
        self,
        query_vector: list[float],
        aspect: str = "problem",
        max_results: int = 10,
        theory_filter: str = "",
        sort_by_significance: bool = False,
        min_dependents: int = 0,
    ) -> list[dict[str, Any]]:
        """Search the Lemma Space static and live indices by cosine similarity, optionally re-ranking by significance."""
        results = []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm
            
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
                    # score = cosine_similarity + 0.15 * normalized_log_dependents
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
        """Search the Definition Space static and live indices by cosine similarity, optionally re-ranking by significance."""
        results = []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm
            
        # 1. Search Static Definitions
        static_matrix = self.definition_embeddings
        if static_matrix is not None and len(static_matrix) > 0:
            norms = np.linalg.norm(static_matrix, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            norm_matrix = static_matrix / norms
            scores = norm_matrix @ q
            
            for idx, score in enumerate(scores):
                meta = self.definition_metadata[idx]
                if theory_filter and theory_filter.lower() not in meta.get("theory", "").lower():
                    continue
                dep_count = int(meta.get("dependents_count", 0))
                if dep_count < min_dependents:
                    continue
                results.append({
                    "definition": meta,
                    "score": float(score),
                    "source": "static",
                })
                
        # 2. Search Live Definitions
        live_list = self.live_definition_embeddings
        if live_list:
            live_matrix = np.array(live_list, dtype=np.float32)
            norms = np.linalg.norm(live_matrix, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            norm_matrix = live_matrix / norms
            scores = norm_matrix @ q
            
            for idx, score in enumerate(scores):
                meta = self.live_definition_metadata[idx]
                if theory_filter and theory_filter.lower() not in meta.get("theory", "").lower():
                    continue
                dep_count = int(meta.get("dependents_count", 0))
                if dep_count < min_dependents:
                    continue
                results.append({
                    "definition": meta,
                    "score": float(score),
                    "source": "live",
                })
                
        # Apply Epistemic Bias Re-ranking
        if sort_by_significance and results:
            import math
            log_deps = []
            for r in results:
                meta = r["definition"]
                dep_count = int(meta.get("dependents_count", 0))
                log_deps.append(math.log1p(dep_count))
            max_log_dep = max(log_deps) if log_deps else 0.0
            if max_log_dep > 1e-10:
                for r, log_dep in zip(results, log_deps):
                    r["score"] = r["score"] + 0.15 * (log_dep / max_log_dep)
                    
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
