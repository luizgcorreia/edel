"""Numpy-based vector index for Isabelle/AFP lemmas."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


class NumpyRAGIndex:
    """Numpy-based cosine similarity vector index.
    
    Stores four separate aspect embedding matrices (problem/statement,
    method/context, finding/strategy, interpretation/dependencies)
    and supports dynamic run-time additions.
    """

    def __init__(self):
        # Static index data (from AFP build)
        self.metadata: list[dict[str, Any]] = []
        self.embeddings: dict[str, np.ndarray | None] = {
            "problem": None,
            "method": None,
            "finding": None,
            "interpretation": None,
        }
        
        # Live session index data (added at runtime by the agent)
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
        
        # Save metadata as JSON (clean of embedding columns)
        meta_df = pd.DataFrame(self.metadata)
        meta_df.to_parquet(directory / "metadata.parquet", index=False)
        
        # Save embeddings as compressed NPZ
        npz_kwargs = {}
        for aspect, arr in self.embeddings.items():
            if arr is not None:
                npz_kwargs[aspect] = arr
        np.savez_compressed(directory / "embeddings.npz", **npz_kwargs)
        print(f"Saved RAG index to {directory} ({len(self.metadata)} lemmas)")

    def load(self, directory: str | Path):
        """Load the index from a directory."""
        directory = Path(directory)
        if not (directory / "metadata.parquet").exists() or not (directory / "embeddings.npz").exists():
            raise FileNotFoundError(f"RAG index files not found in {directory}")
            
        # Load metadata
        meta_df = pd.read_parquet(directory / "metadata.parquet")
        self.metadata = meta_df.to_dict(orient="records")
        
        # Load embeddings
        with np.load(directory / "embeddings.npz") as data:
            for aspect in self.embeddings.keys():
                if aspect in data:
                    self.embeddings[aspect] = data[aspect]
        print(f"Loaded RAG index from {directory} ({len(self.metadata)} lemmas)")

    def build_from_dataframe(self, df: pd.DataFrame):
        """Build the index from an embedded DataFrame."""
        aspects = ["problem", "method", "finding", "interpretation"]
        
        # Check and filter
        meta_cols = [c for c in df.columns if not c.endswith("_embedding")]
        self.metadata = df[meta_cols].to_dict(orient="records")
        
        for aspect in aspects:
            emb_col = f"{aspect}_embedding"
            if emb_col in df.columns:
                # Embeddings are stored as JSON strings in the dataframe
                embs = []
                for val in df[emb_col]:
                    if isinstance(val, str):
                        embs.append(json.loads(val))
                    else:
                        embs.append(val)
                self.embeddings[aspect] = np.array(embs, dtype=np.float32)

    def add_live_lemma(
        self,
        name: str,
        aspect_text_dict: dict[str, str],
        embeddings_dict: dict[str, list[float]],
        theory: str,
        file: str = "",
        line: int = 0,
        proof_text: str = "",
        dependencies: list[str] = [],
    ):
        """Add a newly proven lemma to the in-memory session index."""
        lemma_id = f"{theory}.{name}" if name else f"{theory}.live_lemma_{len(self.live_metadata)}"
        
        record = {
            "title": lemma_id,
            "problem": aspect_text_dict.get("problem", ""),
            "method": aspect_text_dict.get("method", ""),
            "finding": aspect_text_dict.get("finding", ""),
            "interpretation": aspect_text_dict.get("interpretation", ""),
            "theory": theory,
            "file": file,
            "line": line,
            "proof_text": proof_text,
            "statement_text": aspect_text_dict.get("problem", ""),
            "dependencies": ", ".join(dependencies),
            "source": "live",
        }
        
        self.live_metadata.append(record)
        
        # Store embeddings
        for aspect in ["problem", "method", "finding", "interpretation"]:
            emb = embeddings_dict.get(aspect)
            if emb:
                self.live_embeddings[aspect].append(emb)

    def persist_live_lemmas(self, directory: str | Path):
        """Merge live session lemmas into the static index and save to disk."""
        if not self.live_metadata:
            return
            
        # 1. Append live metadata to static metadata (clean of the 'source' key)
        cleaned_live_meta = []
        for meta in self.live_metadata:
            cleaned = meta.copy()
            cleaned.pop("source", None)
            cleaned_live_meta.append(cleaned)
            
        self.metadata.extend(cleaned_live_meta)
        
        # 2. Append live embeddings to static embedding matrices
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
                
        # 3. Clear live data
        self.live_metadata = []
        for aspect in self.live_embeddings:
            self.live_embeddings[aspect] = []
            
        # 4. Save to disk
        self.save(directory)

    def search(
        self,
        query_vector: list[float],
        aspect: str = "problem",
        max_results: int = 10,
        theory_filter: str = "",
    ) -> list[dict[str, Any]]:
        """Search the static and live indices by cosine similarity.
        
        Args:
            query_vector: 1D query embedding list
            aspect: 'problem' (statement), 'method' (context), 'finding' (strategy),
                    'interpretation' (dependencies), or 'all'
            max_results: Max hits to return
            theory_filter: Substring filter on theory name (e.g. 'Multiset')
        """
        results = []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm
            
        # 1. Search Static Index
        static_matrix = self.embeddings.get(aspect)
        if static_matrix is not None and len(static_matrix) > 0:
            # Row-wise norms
            norms = np.linalg.norm(static_matrix, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            norm_matrix = static_matrix / norms
            
            # Cosine similarity
            scores = norm_matrix @ q
            
            for idx, score in enumerate(scores):
                meta = self.metadata[idx]
                if theory_filter and theory_filter.lower() not in meta.get("theory", "").lower():
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
                results.append({
                    "lemma": meta,
                    "score": float(score),
                    "source": "live",
                })
                
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
