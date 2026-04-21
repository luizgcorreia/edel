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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_experiment(name: str, config: dict) -> None:
    """Register a named experiment config.

    Args:
        name: Unique experiment identifier (e.g. "scientometrics_baseline").
        config: Full pipeline config dict (merged over RUN_CONFIG defaults).
    """
    _REGISTRY[name] = copy.deepcopy(config)


def get_experiment(name: str) -> dict:
    """Retrieve a registered experiment config by name.

    Returns a deep copy so callers cannot mutate the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Experiment '{name}' not found. Available: {list(_REGISTRY)}")
    return copy.deepcopy(_REGISTRY[name])


def list_experiments() -> list[str]:
    """Return sorted list of registered experiment names."""
    return sorted(_REGISTRY.keys())


def load_from_file(path: str | Path) -> list[str]:
    """Bulk-register experiments from a JSON config file.

    The file must be a JSON object mapping experiment names to config dicts:
    {
        "my_experiment": { ...full pipeline config... },
        "another_experiment": { ... }
    }

    Each config is deep-merged over RUN_CONFIG defaults so you only need to
    specify the keys that differ.

    Returns the list of registered experiment names.
    """
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


# ---------------------------------------------------------------------------
# Built-in experiment definitions
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> dict:
    """Build a config deep-merged over RUN_CONFIG defaults."""
    return _deep_merge(copy.deepcopy(RUN_CONFIG), overrides)


# ── Baseline: Scientometrics (OpenAlex, ada-002, diffusion) ─────────────────
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
    ),
)

# ── Null model: SCIGen (random computer-science-flavoured papers) ────────────
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
    ),
)

# ── Ablation: different embedding model (small-3) ───────────────────────────
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
    ),
)

# ── Ablation: different LLM for structuring (gpt-4o) ────────────────────────
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
    ),
)

# ── Ablation: UMAP projection ────────────────────────────────────────────────
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
    ),
)


# The public `EXPERIMENTS` view (read-only copy)
EXPERIMENTS: dict[str, dict] = {k: get_experiment(k) for k in list_experiments()}
