"""Experiment registry — named experiment config definitions.

An experiment is a full pipeline config. Any config axis can be varied:
    - data provider / topic
    - LLM model (structuring, labeling)
    - embedding model / dimensions
    - projection method
    - n_documents
    - ...

Use register_experiment() to add new experiments, or load_from_file() to
bulk-register from a JSON/YAML config grid.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from edel.config.defaults import RUN_CONFIG


# ---------------------------------------------------------------------------
# Internal registry store
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict] = {}
_REGISTRY_PATH: Path | None = None

def init_registry(storage_dir: str | Path) -> None:
    """Initialize registry from a persistent JSON file."""
    global _REGISTRY_PATH
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH = storage_dir / "registry.json"
    
    if _REGISTRY_PATH.exists():
        try:
            with _REGISTRY_PATH.open("r") as f:
                _REGISTRY.clear()
                _REGISTRY.update(json.load(f))
                print(f"📂 Loaded {len(_REGISTRY)} experiments from {_REGISTRY_PATH}")
        except Exception as e:
            print(f"Error loading registry: {e}")
            _register_builtins() # Fallback
    else:
        # First run: register built-ins and save
        _register_builtins()
        _save_registry()
        print(f"✨ Initialized registry with built-in experiments at {_REGISTRY_PATH}")
    
    print(f"✅ Registry initialized with {len(_REGISTRY)} experiments.")


def _register_builtins():
    """Register the hardcoded baseline experiments."""
    register_experiment(
        "scientometrics_baseline",
        _make_config(
            data={
                "provider": {
                    "type": "openalex",
                    "topic_id": "T10102",
                    "topic_name": "Scientometrics",
                    "region": None,
                    "params": {"n_documents": 300, "avg_length": 150},
                },
                "transforms": [],
            },
            structured_abstracts={"provider": "openai", "model": "gpt-4o-mini"},
            embedding={"model": "text-embedding-ada-002", "n_dimensions": 1536},
            dimensionality_reduction={"method": "diffusion"},
        )
    )
    register_experiment(
        "scigen_null",
        _make_config(
            data={
                "provider": {
                    "type": "scigen",
                    "topic_name": "SCIGen Null",
                    "params": {"n_documents": 300},
                },
                "transforms": [],
            },
            structured_abstracts={"provider": "openai", "model": "gpt-4o-mini"},
            embedding={"model": "text-embedding-ada-002", "n_dimensions": 1536},
            dimensionality_reduction={"method": "diffusion"},
        )
    )
    register_experiment(
        "scientometrics_small3",
        _make_config(
            data={
                "provider": {
                    "type": "openalex",
                    "topic_id": "T10102",
                    "topic_name": "Scientometrics",
                    "region": None,
                    "params": {"n_documents": 300, "avg_length": 150},
                },
                "transforms": [],
            },
            structured_abstracts={"provider": "openai", "model": "gpt-4o-mini"},
            embedding={"model": "text-embedding-3-small", "n_dimensions": 1536},
            dimensionality_reduction={"method": "diffusion"},
        )
    )
    register_experiment(
        "scientometrics_gpt4o",
        _make_config(
            data={
                "provider": {
                    "type": "openalex",
                    "topic_id": "T10102",
                    "topic_name": "Scientometrics",
                    "region": None,
                    "params": {"n_documents": 300, "avg_length": 150},
                },
                "transforms": [],
            },
            structured_abstracts={"provider": "openai", "model": "gpt-4o"},
            embedding={"model": "text-embedding-ada-002", "n_dimensions": 1536},
            dimensionality_reduction={"method": "diffusion"},
        )
    )
    register_experiment(
        "scientometrics_umap",
        _make_config(
            data={
                "provider": {
                    "type": "openalex",
                    "topic_id": "T10102",
                    "topic_name": "Scientometrics",
                    "region": None,
                    "params": {"n_documents": 300, "avg_length": 150},
                },
                "transforms": [],
            },
            structured_abstracts={"provider": "openai", "model": "gpt-4o-mini"},
            embedding={"model": "text-embedding-ada-002", "n_dimensions": 1536},
            dimensionality_reduction={"method": "umap"},
        )
    )


def _save_registry():
    """Write current registry to disk."""
    if _REGISTRY_PATH:
        try:
            with _REGISTRY_PATH.open("w") as f:
                json.dump(_REGISTRY, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}")


def register_experiment(name: str, config: dict) -> None:
    """Register a named experiment config and persist the registry."""
    _REGISTRY[name] = copy.deepcopy(config)
    _save_registry()

    # Update public view
    if "EXPERIMENTS" in globals():
        EXPERIMENTS[name] = copy.deepcopy(_REGISTRY[name])


def delete_experiment(name: str) -> None:
    """Delete a registered experiment config and update persistence."""
    if name in _REGISTRY:
        del _REGISTRY[name]
        _save_registry()







def get_experiment(name: str) -> dict:
    """Retrieve a registered experiment config by name.

    Returns a deep copy so callers cannot mutate the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Experiment '{name}' not found. Available: {list(_REGISTRY.keys())}")
    return copy.deepcopy(_REGISTRY[name])


def list_experiments() -> list[str]:
    """Return sorted list of registered experiment names."""
    return sorted(_REGISTRY.keys())


def load_from_file(path: str | Path) -> list[str]:
    """Bulk-register experiments from a JSON config file."""
    path = Path(path)
    with path.open("r") as f:
        entries: dict[str, dict] = json.load(f)

    registered = []
    for name, overrides in entries.items():
        config = _deep_merge(copy.deepcopy(RUN_CONFIG), overrides)
        register_experiment(name, config)
        registered.append(name)

    return registered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base, returning merged dict."""
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _make_config(**overrides: Any) -> dict:
    """Build a config deep-merged over RUN_CONFIG defaults."""
    return _deep_merge(copy.deepcopy(RUN_CONFIG), overrides)
