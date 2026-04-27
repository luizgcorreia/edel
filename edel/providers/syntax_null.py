"""Syntax-null synthetic provider adapted from the original pipeline."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from wordfreq import top_n_list, zipf_frequency

from edel.providers.base import (
    ensure_schema,
    generate_random_sentence,
    sample_abstract_length,
)

GRAMMAR_TEMPLATES = [
    # --- Simple clauses ---
    ["DET", "ADJ", "NOUN", "VERB", "DET", "ADJ", "NOUN"],
    ["DET", "NOUN", "VERB", "PREP", "DET", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "NOUN", "PREP", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "ADJ", "NOUN"],
    # --- Starting without determiner ---
    ["ADJ", "NOUN", "VERB", "DET", "NOUN"],
    ["NOUN", "VERB", "DET", "ADJ", "NOUN"],
    ["NOUN", "VERB", "PREP", "DET", "NOUN"],
    ["ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
    # --- Slightly longer academic-like sentences ---
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "ADJ", "NOUN"],
    ["DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "PREP", "NOUN"],
    ["ADJ", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"],
    # --- With adverbs ---
    ["DET", "ADJ", "NOUN", "ADV", "VERB", "DET", "NOUN"],
    ["NOUN", "ADV", "VERB", "DET", "ADJ", "NOUN"],
    ["DET", "NOUN", "VERB", "ADV", "PREP", "DET", "NOUN"],
    # --- Passive-ish structure ---
    ["DET", "NOUN", "VERB", "PREP", "DET", "NOUN", "PREP", "DET", "NOUN"],
    ["ADJ", "NOUN", "VERB", "DET", "ADJ", "NOUN", "PREP", "NOUN"],
    # --- Longer abstract-style sentences ---
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "ADJ", "NOUN", "PREP", "NOUN"],
    ["ADJ", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN", "PREP", "DET", "NOUN"],
]

FUNCTION_WORDS = {
    "DET": ["the", "a", "this", "that"],
    "PREP": ["of", "for", "with", "in", "under", "between", "through"],
}


def build_pos_lexicon(n: int = 100000, min_zipf: int = 3) -> dict[str, list[str]]:
    """Build a part-of-speech lexicon using WordNet and wordfreq."""
    freq_words = {
        w.lower()
        for w in top_n_list("en", n=n)
        if zipf_frequency(w, "en") > min_zipf and w.isalpha() and w.islower() and len(w) > 3
    }

    def wn_pos(p):
        return {l.name().lower() for s in wn.all_synsets(p) for l in s.lemmas()}

    nouns = freq_words & wn_pos("n")
    verbs = freq_words & wn_pos("v")
    adjs = freq_words & wn_pos("a")
    advs = freq_words & wn_pos("r")

    return {
        "NOUN": sorted(nouns),
        "VERB": sorted(verbs),
        "ADJ": sorted(adjs),
        "ADV": sorted(advs),
    }


def generate_grammar_sentence(pos_lexicon: dict[str, list[str]]) -> str:
    """Generate a sentence following a random grammar template."""
    weights = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1]
    template = random.choices(GRAMMAR_TEMPLATES, weights=weights)[0]
    sentence = []

    for slot in template:
        if slot == "VERB":
            word = random.choice(pos_lexicon["VERB"]) + "s"
        elif slot in pos_lexicon:
            word = random.choice(pos_lexicon[slot])
        else:
            word = random.choice(FUNCTION_WORDS[slot])
        sentence.append(word)

    sentence[0] = sentence[0].capitalize()
    return " ".join(sentence)


def generate_grammar_abstract(
    mu: float, sigma: float, pos_lexicon: dict[str, list[str]]
) -> str:
    """Generate a multi-sentence abstract with grammar patterns."""
    target_tokens = sample_abstract_length(mu, sigma)
    sentences = []
    token_count = 0

    while token_count < target_tokens:
        s = generate_grammar_sentence(pos_lexicon)
        sentences.append(s)
        token_count += len(s.split())

    return ". ".join(sentences) + "."


def generate_dataset(config: dict) -> tuple[pd.DataFrame, dict]:
    """Generate a synthetic syntax-null dataset with complex grammar patterns."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})

    n_docs = params.get("n_documents", 10)
    mu = params.get("abstract_length_mu", 6.7)
    sigma = params.get("abstract_length_sigma", 0.6)
    seed = params.get("seed")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    pos_lexicon = build_pos_lexicon()
    records = []

    for i in range(n_docs):
        title = generate_grammar_sentence(pos_lexicon)
        abstract = generate_grammar_abstract(mu, sigma, pos_lexicon)
        keywords = generate_random_sentence(6, pos_lexicon["NOUN"], join=False)

        records.append(
            {
                "source_provider": "syntax_null",
                "id": f"syntax_null_{i}",
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

    return ensure_schema(pd.DataFrame(records), provider_name="syntax_null"), {}
