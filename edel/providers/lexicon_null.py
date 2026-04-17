"""Lexicon-null synthetic provider."""

from __future__ import annotations

import random

import pandas as pd

from edel.providers.base import ensure_schema


LEXICON = [
    "analysis",
    "model",
    "theory",
    "evidence",
    "method",
    "network",
    "learning",
    "science",
    "dataset",
    "citation",
]


def _sentence(rng: random.Random, n_tokens: int) -> str:
    return " ".join(rng.choice(LEXICON) for _ in range(n_tokens)).capitalize() + "."


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a synthetic lexicon-null dataset with randomized token strings."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    n_docs = int(params.get("n_documents", 5))
    seed = int(params.get("seed", 0))

    rng = random.Random(seed)
    records = []

    for i in range(n_docs):
        title = _sentence(rng, 6)
        abstract = " ".join(_sentence(rng, 10) for _ in range(3))
        records.append(
            {
                "source_provider": "lexicon_null",
                "id": f"lexicon_null:{i}",
                "title": title,
                "abstract": abstract,
                "authorships": [],
                "publication_year": None,
                "cited_by_count": 0,
                "citation_normalized_percentile": 0.0,
                "doi": None,
                "oa_status": None,
                "primary_location": None,
                "countries": [],
                "topics": [],
                "type": "synthetic",
                "language": "en",
                "keywords": [rng.choice(LEXICON) for _ in range(4)],
                "has_fulltext": False,
            }
        )

    return ensure_schema(pd.DataFrame(records), provider_name="lexicon_null")
