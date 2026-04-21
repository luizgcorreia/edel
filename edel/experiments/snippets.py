"""Persistent storage and management for configuration snippets used in parameter sweeps."""

from __future__ import annotations
import json
import copy
from pathlib import Path
from typing import Any

_SNIPPETS: dict[str, dict[str, dict]] = {}
_SNIPPETS_PATH: Path | None = None

# Mapping of stage display names to RUN_CONFIG keys
# We use an OrderedDict-like approach to ensure consistent order
STAGE_KEYS = {
    "Structured Abstracts": "structured_abstracts",
    "Embeddings": "embedding",
    "Projection": "dimensionality_reduction",
    "Vector Field": "vector_field",
    "Clustering": "clustering",
    "Labeling": "labeling",
    "Landscape": "landscape",
}

# Canonical order of stages for UI and callbacks
STAGE_LIST = list(STAGE_KEYS.keys())

def init_snippets(storage_dir: str | Path) -> None:
    """Initialize snippets from a persistent JSON file."""
    global _SNIPPETS_PATH
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    _SNIPPETS_PATH = storage_dir / "snippets.json"
    
    if _SNIPPETS_PATH.exists():
        try:
            with _SNIPPETS_PATH.open("r") as f:
                _SNIPPETS.clear()
                _SNIPPETS.update(json.load(f))
                print(f"📂 Loaded snippets for {len(_SNIPPETS)} stages from {_SNIPPETS_PATH}")
        except Exception as e:
            print(f"Error loading snippets: {e}")
            _load_defaults()
    else:
        _load_defaults()
        _save_snippets()
        print(f"✨ Initialized snippets with defaults at {_SNIPPETS_PATH}")

def _load_defaults():
    """Load some initial helpful snippets."""
    _SNIPPETS.clear()
    
    # Embeddings
    _SNIPPETS["Embeddings"] = {
        "ada-002": {"model": "text-embedding-ada-002", "n_dimensions": 1536},
        "3-small": {"model": "text-embedding-3-small", "n_dimensions": 1536},
        "3-large": {"model": "text-embedding-3-large", "n_dimensions": 3072},
    }
    
    # Projection
    _SNIPPETS["Projection"] = {
        "diffusion": {"method": "diffusion"},
        "umap": {"method": "umap", "n_neighbors": 15, "min_dist": 0.1},
    }
    
    # Clustering
    _SNIPPETS["Clustering"] = {
        "hdbscan_small": {"domain": {"algorithm": "hdbscan", "params": {"min_cluster_size": 10}}},
        "hdbscan_large": {"domain": {"algorithm": "hdbscan", "params": {"min_cluster_size": 30}}},
    }

def _save_snippets():
    """Write current snippets to disk."""
    if _SNIPPETS_PATH:
        try:
            with _SNIPPETS_PATH.open("w") as f:
                json.dump(_SNIPPETS, f, indent=2)
        except Exception as e:
            print(f"Error saving snippets: {e}")

def get_snippets(stage: str) -> dict[str, dict]:
    """Return all snippets for a given stage."""
    return copy.deepcopy(_SNIPPETS.get(stage, {}))

def save_snippet(stage: str, name: str, config: dict) -> None:
    """Save a new snippet for a stage."""
    if stage not in _SNIPPETS:
        _SNIPPETS[stage] = {}
    _SNIPPETS[stage][name] = copy.deepcopy(config)
    _save_snippets()

def delete_snippet(stage: str, name: str) -> None:
    """Remove a snippet from a stage."""
    if stage in _SNIPPETS and name in _SNIPPETS[stage]:
        del _SNIPPETS[stage][name]
        _save_snippets()
