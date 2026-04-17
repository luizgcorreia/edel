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


def sample_abstract_length(mu: float, sigma: float) -> int:
    """Sample abstract length from lognormal distribution."""
    import numpy as np

    return int(max(200, min(np.random.lognormal(mu, sigma), 4000)))


def load_lexicon(lexicon_file: str | None = None) -> list[str]:
    """Load or generate common English lexicon."""
    if lexicon_file:
        with open(lexicon_file, "r") as f:
            return f.read().splitlines()

    from wordfreq import top_n_list, zipf_frequency

    vocab = [w for w in top_n_list("en", n=100000) if zipf_frequency(w, "en") > 3]

    # basic cleanup
    vocab = [w for w in vocab if w.isalpha() and len(w) > 2]

    return vocab


def generate_random_sentence(avg_length: int, lexicon: list[str], join: bool = True) -> Any:
    """Generate a random sentence by sampling tokens from lexicon."""
    import numpy as np

    length = np.random.poisson(avg_length)
    if length <= 0:
        length = 1
    words = np.random.choice(lexicon, size=length)
    if join:
        return " ".join(words).capitalize()
    return list(words)
