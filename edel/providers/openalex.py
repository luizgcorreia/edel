"""OpenAlex provider (mock Stage 1 implementation)."""

from __future__ import annotations

import pandas as pd

from edel.providers.base import ensure_schema


def generate_dataset(config: dict) -> pd.DataFrame:
    """Generate a standardized OpenAlex-like dataset from config."""
    provider_cfg = config.get("provider", {})
    params = provider_cfg.get("params", {})
    n_docs = int(params.get("n_documents", 5))
    topic_id = provider_cfg.get("topic_id", "T00000")
    topic_name = provider_cfg.get("topic_name", "Unknown Topic")

    records = []
    for i in range(n_docs):
        records.append(
            {
                "source_provider": "openalex",
                "id": f"https://openalex.org/W{100000 + i}",
                "title": f"Mock OpenAlex paper {i} on {topic_name}",
                "abstract": f"This is a mock abstract for {topic_id} record {i}.",
                "authorships": [],
                "publication_year": 2020 + (i % 5),
                "cited_by_count": i * 3,
                "citation_normalized_percentile": min(1.0, round(i / max(n_docs, 1), 3)),
                "doi": f"10.0000/mock-openalex-{i}",
                "oa_status": "gold",
                "primary_location": "Mock Journal",
                "countries": provider_cfg.get("region") or [],
                "topics": [topic_name],
                "type": "article",
                "language": "en",
                "keywords": params.get("filter_keywords") or ["mock", "openalex"],
                "has_fulltext": True,
            }
        )

    return ensure_schema(pd.DataFrame(records), provider_name="openalex")
