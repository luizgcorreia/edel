"""SCIGen-null synthetic provider (mock implementation)."""

from __future__ import annotations

import pandas as pd

from edel.providers.base import ensure_schema


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a SCIGen-like synthetic dataset (mock records only)."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    n_docs = int(params.get("n_documents", 5))

    records = []
    for i in range(n_docs):
        records.append(
            {
                "source_provider": "scigen_null",
                "id": f"scigen:{i}",
                "title": f"Automatically Generated Research Paper {i}",
                "abstract": "This paper proposes a fully synthetic benchmark and reports mock results.",
                "authorships": [],
                "publication_year": 2005 + (i % 10),
                "cited_by_count": 0,
                "citation_normalized_percentile": 0.0,
                "doi": None,
                "oa_status": None,
                "primary_location": "SCIGen",
                "countries": [],
                "topics": ["Synthetic Text"],
                "type": "synthetic",
                "language": "en",
                "keywords": ["scigen", "null", "generator"],
                "has_fulltext": True,
            }
        )

    return ensure_schema(pd.DataFrame(records), provider_name="scigen_null")
