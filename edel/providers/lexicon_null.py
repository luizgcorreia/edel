"""Lexicon-null synthetic provider adapted from the original pipeline."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from wordfreq import top_n_list, zipf_frequency

from edel.providers.base import ensure_schema


def sample_abstract_length(mu: float, sigma: float) -> int:
    """Sample abstract length from lognormal distribution.
    
    Args:
        mu: Mean of the log distribution.
        sigma: Standard deviation of the log distribution.
    """
    return int(max(200, min(np.random.lognormal(mu, sigma), 4000)))


def load_lexicon(lexicon_file: str | None = None) -> list[str]:
    """Load or generate common English lexicon.
    
    Args:
        lexicon_file: Path to a line-separated text file with tokens.
            If None, generates a lexicon from top English words.
    """
    if lexicon_file:
        with open(lexicon_file, "r") as f:
            return f.read().splitlines()

    vocab = [w for w in top_n_list("en", n=100000) if zipf_frequency(w, "en") > 3]

    # basic cleanup
    vocab = [w for w in vocab if w.isalpha() and len(w) > 2]

    return vocab


def generate_random_sentence(avg_length: int, lexicon: list[str], join: bool = True) -> Any:
    """Generate a random sentence by sampling tokens from lexicon.
    
    Args:
        avg_length: Poisson mean for the number of words.
        lexicon: List of available tokens.
        join: If True, returns a capitalized string. Otherwise returns list of words.
    """
    length = np.random.poisson(avg_length)
    if length <= 0:
        length = 1
    words = np.random.choice(lexicon, size=length)
    if join:
        return " ".join(words).capitalize()
    return list(words)


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
                "keywords": keywords,
                "has_fulltext": False,
            }
        )

    return ensure_schema(pd.DataFrame(records), provider_name="lexicon_null")
