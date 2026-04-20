"""Lexicon-null synthetic provider adapted from the original pipeline."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from edel.providers.base import (
    ensure_schema,
    generate_random_sentence,
    load_lexicon,
    sample_abstract_length,
)


def generate_random_abstract(mu: float, sigma: float, lexicon: list[str]) -> str:
    """Generate a random abstract by concatenating random sentences.

    Args:
        mu: Abstract length lognormal mu.
        sigma: Abstract length lognormal sigma.
        lexicon: List of available tokens.
    """
    target_tokens = sample_abstract_length(mu, sigma)

    sentences = []
    token_count = 0

    while token_count < target_tokens:
        s = generate_random_sentence(10, lexicon)
        sentences.append(s)
        token_count += len(s.split())

    return ". ".join(sentences) + "."


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a synthetic lexicon-null dataset with randomized tokens.

    Expected config structure:
    {
        "provider": {
            "params": {
                "n_documents": int,
                "abstract_length_mu": float (default 5.2),
                "abstract_length_sigma": float (default 0.5),
                "lexicon_file": str (optional),
                "seed": int (optional)
            }
        }
    }
    """
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})

    n_docs = params.get("n_documents", 10)
    mu = params.get("abstract_length_mu", 5.2)
    sigma = params.get("abstract_length_sigma", 0.5)
    lexicon_file = params.get("lexicon_file")
    seed = params.get("seed")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    lexicon = load_lexicon(lexicon_file)
    records = []

    for i in range(n_docs):
        title = generate_random_sentence(10, lexicon)
        abstract = generate_random_abstract(mu, sigma, lexicon)
        keywords = generate_random_sentence(6, lexicon, join=False)

        records.append(
            {
                "source_provider": "lexicon_null",
                "id": f"lexicon_null_{i}",
                "title": title,
                "abstract_text": abstract,
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
                "keywords": keywords,
                "has_fulltext": False,
            }
        )

    return ensure_schema(pd.DataFrame(records), provider_name="lexicon_null")
