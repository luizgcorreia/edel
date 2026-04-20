"""OpenAlex data provider using the OpenAlex API."""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from edel.io.openalex import (
    extract_country_codes,
    inverted_index_to_text,
    openalex_request,
)
from edel.providers.base import ensure_schema


def filter_keywords(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """Filter out rows where the abstract contains any of the specified keywords."""
    if not keywords or df.empty:
        return df

    initial_shape = df.shape
    print(f"Filtering non-research works. Shape before: {initial_shape}")

    # Create a regex pattern from the keywords
    pattern = "|".join(keywords)
    mask = df["abstract_text"].str.contains(pattern, case=False, na=False)
    df_filtered = df[~mask]

    print(f"Shape after filtering: {df_filtered.shape}")
    return df_filtered


def generate_dataset(config: dict) -> pd.DataFrame:
    """Harvest works from OpenAlex based on topic and optional region filters.
    
    Expected config structure:
    {
        "provider": {
            "type": "openalex",
            "topic_id": "T10102",
            "region": ["US", "GB"] (optional),
            "params": {
                "n_documents": int,
                "filter_keywords": list[str] (optional)
            }
        }
    }
    """
    provider_cfg = config.get("provider", {})
    topic_id = provider_cfg.get("topic_id")
    region = provider_cfg.get("region")
    params = provider_cfg.get("params", {})
    limit = params.get("n_documents")
    keywords = params.get("filter_keywords")

    if not topic_id:
        raise ValueError("topic_id must be specified in provider config for OpenAlex.")

    # 1. Define filters
    filter_parts = [
        f"topics.id:{topic_id}",
        "has_abstract:true",
        "is_paratext:false",
    ]
    if region:
        # OpenAlex expects pipe-separated country codes for OR logic
        country_filter = "|".join(region)
        filter_parts.append(f"institutions.country_code:{country_filter}")

    filters = ",".join(filter_parts)

    # 2. Harvest Records
    all_records = []
    current_cursor = "*"
    print(f"Starting OpenAlex harvesting for topic {topic_id}...")

    # Initialize progress bar if we have a limit
    pbar = tqdm(total=limit, desc="Harvesting OpenAlex") if limit else None

    try:
        while True:
            # per_page max is 200
            fetch_count = 200
            if limit:
                remaining = limit - len(all_records)
                if remaining <= 0:
                    break
                fetch_count = min(200, remaining)

            data = openalex_request(filters, per_page=fetch_count, cursor=current_cursor)

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                # We need the inverted index to reconstruct the abstract text
                inv_idx = work.get("abstract_inverted_index")
                if not inv_idx:
                    continue

                abstract_text = inverted_index_to_text(inv_idx)

                all_records.append(
                    {
                        "source_provider": "openalex",
                        "id": work.get("id"),
                        "title": work.get("title"),
                        "abstract_text": abstract_text,
                        "authorships": work.get("authorships", []),
                        "publication_year": work.get("publication_year"),
                        "cited_by_count": work.get("cited_by_count"),
                        "citation_normalized_percentile": (
                            work.get("citation_normalized_percentile", {}).get("value")
                            if work.get("citation_normalized_percentile")
                            else 0
                        ),
                        "doi": work.get("doi"),
                        "oa_status": (work.get("open_access") or {}).get("oa_status"),
                        "primary_location": (
                            ((work.get("primary_location") or {}).get("source") or {}).get(
                                "display_name"
                            )
                        ),
                        "countries": extract_country_codes(work.get("authorships", [])),
                        "topics": [
                            t.get("display_name")
                            for t in work.get("topics", [])
                            if t.get("display_name")
                        ],
                        "type": work.get("type"),
                        "language": work.get("language"),
                        "keywords": [
                            k.get("display_name")
                            for k in work.get("keywords", [])
                            if k.get("display_name")
                        ],
                        "has_fulltext": work.get("has_fulltext"),
                    }
                )
                if pbar:
                    pbar.update(1)

                if limit and len(all_records) >= limit:
                    break

            # Handle pagination
            next_cursor = data.get("meta", {}).get("next_cursor")
            if next_cursor and next_cursor != current_cursor:
                current_cursor = next_cursor
            else:
                break

            if limit and len(all_records) >= limit:
                break
    finally:
        if pbar:
            pbar.close()

    if not all_records:
        return ensure_schema(pd.DataFrame(columns=["abstract_text"]), provider_name="openalex")

    df = pd.DataFrame(all_records)

    # 3. Post-harvest filtering
    if keywords:
        df = filter_keywords(df, keywords)

    return ensure_schema(df, provider_name="openalex")
