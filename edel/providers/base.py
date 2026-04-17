"""Base helpers for Stage 1 data providers."""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "source_provider",
    "id",
    "title",
    "abstract",
    "authorships",
    "publication_year",
    "cited_by_count",
    "citation_normalized_percentile",
    "doi",
    "oa_status",
    "primary_location",
    "countries",
    "topics",
    "type",
    "language",
    "keywords",
    "has_fulltext",
]

_DEFAULTS: dict[str, Any] = {
    "source_provider": "unknown",
    "id": None,
    "title": "",
    "abstract": "",
    "authorships": [],
    "publication_year": None,
    "cited_by_count": 0,
    "citation_normalized_percentile": 0.0,
    "doi": None,
    "oa_status": None,
    "primary_location": None,
    "countries": [],
    "topics": [],
    "type": "article",
    "language": "en",
    "keywords": [],
    "has_fulltext": False,
}


def ensure_schema(df: pd.DataFrame, provider_name: str) -> pd.DataFrame:
    """Return a copy of ``df`` with the required provider schema columns present."""
    out = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            value = _DEFAULTS[col]
            out[col] = [value.copy() if isinstance(value, list) else value for _ in range(len(out))]

    if "source_provider" in out.columns:
        out["source_provider"] = out["source_provider"].fillna(provider_name)

    return out[REQUIRED_COLUMNS]
