"""Syntax-null synthetic provider."""

from __future__ import annotations

import random

import pandas as pd

from edel.providers.base import ensure_schema


SUBJECTS = ["Model", "Framework", "Approach", "Method"]
VERBS = ["examines", "evaluates", "describes", "simulates"]
OBJECTS = ["evidence", "citations", "patterns", "structures"]
MODIFIERS = ["systematically", "empirically", "formally", "carefully"]


def _grammar_sentence(rng: random.Random) -> str:
    return f"{rng.choice(SUBJECTS)} {rng.choice(MODIFIERS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}."


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a synthetic syntax-null dataset with simple grammar patterns."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    n_docs = int(params.get("n_documents", 5))
    seed = int(params.get("seed", 1))

    rng = random.Random(seed)
    records = []

    for i in range(n_docs):
        records.append(
            {
                "source_provider": "syntax_null",
                "id": f"syntax_null:{i}",
                "title": _grammar_sentence(rng),
                "abstract": " ".join(_grammar_sentence(rng) for _ in range(4)),
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
                "keywords": ["syntax", "null", "synthetic"],
                "has_fulltext": False,
            }
        )

    return ensure_schema(pd.DataFrame(records), provider_name="syntax_null")
