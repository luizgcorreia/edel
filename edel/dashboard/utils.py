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
    """Convert DataFrame to Dash DataTable records (list of row dicts).
    Ensures that values are JSON-serializable by converting complex types to strings.
    """
    subset = df.head(max_rows).copy()
    # Convert non-serializable objects to strings
    for col in subset.columns:
        if subset[col].dtype == object:
            # Check if any element in the column is a non-standard type
            subset[col] = subset[col].apply(lambda x: str(x) if not isinstance(x, (int, float, str, bool, type(None))) else x)
    return subset.to_dict("records")


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


def parse_authorships(authorships: Any) -> list[dict]:
    """Parse OpenAlex authorships data into a clean list of author metadata."""
    if authorships is None:
        return []
    
    # Handle cases where it might be a JSON string or already a list/array
    data = authorships
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return []
            
    if not hasattr(data, "__iter__") or isinstance(data, (str, dict)):
        return []
        
    parsed = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
            
        author_info = entry.get("author", {})
        author_name = author_info.get("display_name", entry.get("raw_author_name", "Unknown"))
        author_id = author_info.get("id", "")
        author_orcid = author_info.get("orcid", "")
        
        # Institutions
        institutions = []
        inst_data = entry.get("institutions", [])
        # Handle both lists and numpy arrays
        if hasattr(inst_data, "__iter__") and not isinstance(inst_data, (str, dict)):
            for inst in inst_data:
                if isinstance(inst, dict):
                    institutions.append({
                        "name": inst.get("display_name", "Unknown Institution"),
                        "country": inst.get("country_code", ""),
                        "id": inst.get("id", "")
                    })
        
        parsed.append({
            "name": author_name,
            "id": author_id,
            "orcid": author_orcid,
            "position": entry.get("author_position", ""),
            "is_corresponding": entry.get("is_corresponding", False),
            "institutions": institutions
        })
        
    return parsed
