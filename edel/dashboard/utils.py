"""Shared utilities for the EDEL dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from edel.experiments.registry import list_experiments, get_experiment


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_registry_options() -> list[dict]:
    """Return Dash dropdown options for all registered experiment configs."""
    return [{"label": name, "value": name} for name in list_experiments()]


def config_to_json(name: str) -> str:
    """Return pretty-printed JSON of a registered config."""
    try:
        return json.dumps(get_experiment(name), indent=2)
    except KeyError:
        return "{}"


def parse_config_json(text: str) -> dict | None:
    """Parse JSON text into a config dict; return None on error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Artifact browser helpers
# ---------------------------------------------------------------------------

def scan_artifact_stages(base_path: str | Path) -> list[str]:
    """Return stage directories that exist under base_path."""
    from edel.io.artifact import CANONICAL_STAGE_NAMES
    base_path = Path(base_path)
    return [s for s in CANONICAL_STAGE_NAMES if (base_path / s).exists()]


# ---------------------------------------------------------------------------
# DataFrame display helpers
# ---------------------------------------------------------------------------

def df_to_dash_columns(df: pd.DataFrame, max_cols: int = 30) -> list[dict]:
    """Convert DataFrame columns to Dash DataTable column descriptors."""
    cols = list(df.columns)[:max_cols]
    return [{"name": c, "id": c} for c in cols]


def df_to_dash_records(df: pd.DataFrame, max_rows: int = 200) -> list[dict]:
    """Convert DataFrame to Dash DataTable records (list of row dicts)."""
    return df.head(max_rows).to_dict("records")


# ---------------------------------------------------------------------------
# Nested config accessor
# ---------------------------------------------------------------------------

def get_nested(d: dict, dotted_path: str, default: Any = None) -> Any:
    """Safely read a dotted path from a nested dict."""
    keys = dotted_path.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
