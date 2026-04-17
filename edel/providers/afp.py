"""AFP provider (mock Stage 1 implementation)."""

from __future__ import annotations

import pandas as pd

from edel.providers.base import ensure_schema


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a standardized AFP-like dataset from config."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    n_docs = int(params.get("n_documents", 5))

    records = []
    for i in range(n_docs):
        records.append(
            {
                "source_provider": "afp",
                "id": f"afp:{i}",
                "title": f"Mock AFP theory {i}",
                "abstract": f"Formal proof artifact {i} extracted from AFP mock corpus.",
                "authorships": [],
                "publication_year": 2018 + (i % 7),
                "cited_by_count": 0,
                "citation_normalized_percentile": 0.0,
                "doi": None,
                "oa_status": "closed",
                "primary_location": "Archive of Formal Proofs",
                "countries": [],
                "topics": ["Formal Methods"],
                "type": "theory",
                "language": "en",
                "keywords": ["afp", "isabelle"],
                "has_fulltext": True,
            }
        )

    return ensure_schema(pd.DataFrame(records), provider_name="afp")
