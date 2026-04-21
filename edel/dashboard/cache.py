"""Results cache for the EDEL dashboard.

Precomputes and persists experiments/results.parquet so that
dashboard callbacks never call analyze_experiments() directly.

Delta rebuild: only re-analyzes experiments whose artifact
mtime is newer than the cache file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from edel.experiments.analyzer import analyze_experiments
from edel.experiments.runner import load_registry

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "results.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_results_df(base_path: str | Path = "artifacts") -> pd.DataFrame:
    """Load the precomputed results DataFrame.

    Returns an empty DataFrame if the cache does not exist yet.
    """
    cache_path = _cache_path(base_path)
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(cache_path)
    except Exception as e:
        logger.warning(f"Failed to load results cache: {e}")
        return pd.DataFrame()


def rebuild_results_cache(
    base_path: str | Path = "artifacts",
    delta_only: bool = True,
) -> pd.DataFrame:
    """Rebuild (or delta-update) results.parquet from experiment artifacts.

    Args:
        base_path: Root artifact directory.
        delta_only: If True (default), only re-analyze experiments whose
            landscape artifact is newer than the cache. If False, rebuild
            from scratch.

    Returns:
        The full (updated) results DataFrame.
    """
    base_path = Path(base_path)
    cache_path = _cache_path(base_path)

    registry = load_registry(base_path)
    if not registry:
        logger.info("No experiments in registry — cache is empty.")
        return pd.DataFrame()

    # ── Delta logic ──────────────────────────────────────────────────────────
    if delta_only and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        stale = _find_stale_experiments(registry, base_path, cache_mtime)

        if not stale:
            logger.info("Cache is up to date — no delta needed.")
            return get_results_df(base_path)

        logger.info(f"Delta rebuild: {len(stale)} experiment(s) to re-analyze.")
        existing_df = get_results_df(base_path)

        new_rows = analyze_experiments(stale, base_path=base_path)

        # Upsert: replace existing rows for re-analyzed experiments
        if not existing_df.empty and "experiment_id" in existing_df.columns:
            new_ids = set(new_rows["experiment_id"])
            existing_df = existing_df[~existing_df["experiment_id"].isin(new_ids)]
            full_df = pd.concat([existing_df, new_rows], ignore_index=True)
        else:
            full_df = new_rows

    else:
        # Full rebuild
        logger.info(f"Full cache rebuild: {len(registry)} experiment(s).")
        full_df = analyze_experiments(registry, base_path=base_path)

    # ── Persist ───────────────────────────────────────────────────────────────
    if not full_df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(cache_path, index=False)
        logger.info(f"Cache saved: {cache_path} ({len(full_df)} rows)")
    else:
        logger.warning("No results to cache (analysis returned empty DataFrame).")

    return full_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(base_path: str | Path) -> Path:
    return Path(base_path) / "experiments" / _CACHE_FILENAME


def _find_stale_experiments(
    registry: list[dict],
    base_path: Path,
    cache_mtime: float,
) -> list[dict]:
    """Return experiments whose landscape artifact is newer than the cache."""
    stale = []
    for exp in registry:
        refs = exp.get("artifact_refs", {})
        landscape_art = refs.get("landscape")
        if landscape_art is None:
            stale.append(exp)  # no artifact yet → always stale
            continue

        # Check both parquet and pkl variants
        for p in [landscape_art.parquet_path, landscape_art.pkl_path]:
            if p.exists() and p.stat().st_mtime > cache_mtime:
                stale.append(exp)
                break

    return stale
