"""Explicit provider registry for Stage 1 data collection."""

from edel.providers import afp, lexicon_null, openalex, scigen_null, syntax_null

PROVIDERS = {
    "openalex": openalex.generate_dataset,
    "afp": afp.generate_dataset,
    "lexicon_null": lexicon_null.generate_dataset,
    "syntax_null": syntax_null.generate_dataset,
    "scigen_null": scigen_null.generate_dataset,
}


def get_provider(name: str):
    """Return provider callable by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")
    return PROVIDERS[name]
