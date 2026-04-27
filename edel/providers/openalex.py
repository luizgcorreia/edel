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


def process_work_json(work: dict) -> dict | None:
    """Extract and format fields from OpenAlex Work JSON."""
    inv_idx = work.get("abstract_inverted_index")
    if not inv_idx:
        return None

    abstract_text = inverted_index_to_text(inv_idx)

    return {
        "source_provider": "openalex",
        "id": work.get("id"),
        "title": work.get("title"),
        "abstract_text": abstract_text,
        "authorships": work.get("authorships", []),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "citation_normalized_percentile": (work.get("citation_normalized_percentile", {}).get("value") if work.get("citation_normalized_percentile") else 0),
        "doi": work.get("doi"),
        "oa_status": (work.get("open_access") or {}).get("oa_status"),
        "primary_location": (((work.get("primary_location") or {}).get("source") or {}).get("display_name")),
        "countries": extract_country_codes(work.get("authorships", [])),
        "topics": [t.get("display_name") for t in work.get("topics", []) if t.get("display_name")],
        "type": work.get("type"),
        "language": work.get("language"),
        "keywords": [k.get("display_name") for k in work.get("keywords", []) if k.get("display_name")],
        "has_fulltext": work.get("has_fulltext"),
    }


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
    sampling_strategy = params.get("sampling_strategy", "probabilistic")
    sampling_seed = params.get("sampling_seed")

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

    # Pre-flight request to get total count
    try:
        initial_data = openalex_request(filters, per_page=1, cursor="*")
        total_count = initial_data.get("meta", {}).get("count", 0)
    except Exception as e:
        print(f"Failed to fetch initial metadata from OpenAlex: {e}")
        total_count = 0

    if limit is None:
        limit = total_count
        print(f"No limit specified. Attempting to harvest all {limit} documents.")
    else:
        limit = min(limit, total_count)
        print(f"Harvesting up to {limit} documents (total available: {total_count}).")

    # 2. Harvest Records
    all_records = []
    seen_ids = set()
    
    print(f"Starting OpenAlex harvesting for topic {topic_id}...")
    pbar = tqdm(total=limit, desc="Harvesting OpenAlex") if limit > 0 else None

    try:
        # Cursor-based harvesting for both probabilistic and deterministic modes
        current_cursor = "*"
        
        # Only sort by citation if deterministic
        sort_param = "cited_by_count:desc" if (limit and sampling_strategy == "deterministic") else None
        
        while True:
            fetch_count = 200
            if limit:
                remaining = limit - len(all_records)
                if remaining <= 0:
                    break
                fetch_count = min(200, remaining)

            try:
                data = openalex_request(filters, per_page=fetch_count, cursor=current_cursor, sort=sort_param)
            except Exception as e:
                print(f"\nError fetching page (cursor: {current_cursor}): {e}. Retrying in 5s...")
                import time
                time.sleep(5)
                continue

            page_results = data.get("results", [])
            if not page_results:
                break
            
            for work in page_results:
                if work.get("id") in seen_ids:
                    continue
                
                processed = process_work_json(work)
                if processed:
                    seen_ids.add(processed["id"])
                    all_records.append(processed)
                    if pbar:
                        pbar.update(1)

                if limit and len(all_records) >= limit:
                    break

            next_cursor = data.get("meta", {}).get("next_cursor")
            if next_cursor and next_cursor != current_cursor:
                current_cursor = next_cursor
            else:
                break

            if limit and len(all_records) >= limit:
                break

    finally:
        pbar.close()

    if not all_records:
        return ensure_schema(pd.DataFrame(columns=["abstract_text"]), provider_name="openalex")

    df = pd.DataFrame(all_records)
    
    # Optional local shuffle for probabilistic to randomize the ordered cursor results
    if sampling_strategy == "probabilistic":
        seed = config.get("random_seed") or sampling_seed or 42
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 3. Post-harvest filtering
    if keywords:
        df = filter_keywords(df, keywords)

    return ensure_schema(df, provider_name="openalex")
