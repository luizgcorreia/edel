"""OpenAlex API helpers."""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.openalex.org/"


def openalex_request(
    filters: str, per_page: int = 200, cursor: str = "*", endpoint: str = "works"
) -> dict[str, Any]:
    """Execute a request against the OpenAlex API."""
    api_key = os.environ.get("OPENALEX_API_KEY")

    params = {
        "filter": filters,
        "per-page": per_page,
        "cursor": cursor,
    }
    if api_key:
        params["api_key"] = api_key

    response = requests.get(urllib.parse.urljoin(BASE_URL, endpoint), params=params)

    if response.status_code != 200:
        raise RuntimeError(f"OpenAlex error {response.status_code}: {response.text}")

    return response.json()


def extract_country_codes(authorships: list[dict[str, Any]]) -> list[str]:
    """Extract unique country codes from authorships metadata."""
    countries = set()
    for author in authorships:
        institutions = author.get("institutions", [])
        for inst in institutions:
            country = inst.get("country_code")
            if country:
                countries.add(country)
    return sorted(list(countries))


def inverted_index_to_text(inv_idx: dict[str, list[int]]) -> str:
    """Convert OpenAlex inverted abstract into plain text."""
    if not inv_idx:
        return ""
    max_pos = max(pos for positions in inv_idx.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inv_idx.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)
