"""Runner — batch execution engine.

Phase 1 of the experiments engine: data generation only.

    configs → run_full_pipeline() → artifacts → experiment registry

Does NOT compute metrics. Saves an experiment registry (list of records)
at {base_path}/experiments/registry.pkl.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from edel.io.artifact import (
    Artifact,
    get_experiment_label,
    make_stage_artifact,
    save_artifact,
)
from edel.pipeline.run import run_full_pipeline

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "registry.pkl"

# Canonical stage → artifact name mapping (mirrors run_full_pipeline internals)
_STAGE_ARTIFACT_MAP: list[tuple[str, str, str]] = [
    ("data",         "data_collection",       "dataset"),
    ("structuring",  "structured_abstracts",   "sa"),
    ("embedding",    "embeddings",             "embeddings"),
    ("projection",   "dimensionality_reduction", "dr"),
    ("vector_field", "vector_field",           "vf"),
    ("clustering",   "clustering",             "clustering"),
    ("labels",       "labeling",               "labeled"),
    ("landscape",    "output",                 "landscape_results"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiments(
    configs: list[dict],
    base_path: str | Path = "artifacts",
    force: bool = False,
) -> list[dict]:
    """Run a list of pipeline configs and record artifact references.

    For each config:
        1. Delegates to run_full_pipeline() — saves all stage artifacts (existing system).
        2. Re-derives artifact descriptors from config (deterministic addressing).
        3. Builds an experiment record: {experiment_id, config, artifact_refs}.

    The registry is saved to {base_path}/experiments/registry.pkl (upsert by
    experiment_id so re-runs update entries without losing others).

    Args:
        configs: List of full pipeline config dicts.
        base_path: Root artifact directory.
        force: If True, forces recomputation of all pipeline stages.

    Returns:
        List of experiment records.
    """
    base_path = Path(base_path)
    records = []

    for i, config in enumerate(configs):
        experiment_id = get_experiment_label(config)
        logger.info(f"[{i+1}/{len(configs)}] Running experiment: {experiment_id}")
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(configs)}: {experiment_id}")
        print(f"{'='*60}")

        try:
            # Phase 1: run pipeline (saves all stage artifacts)
            run_full_pipeline(config, base_path=base_path, force=force)

            # Build artifact refs by re-deriving from config (always deterministic)
            artifact_refs = _build_artifact_refs(config, base_path)

            record = {
                "experiment_id": experiment_id,
                "config": config,
                "artifact_refs": artifact_refs,
            }
            records.append(record)
            print(f"✅ Experiment complete: {experiment_id}")

        except Exception as e:
            logger.error(f"Experiment '{experiment_id}' failed: {e}", exc_info=True)
            print(f"❌ Experiment failed: {experiment_id} — {e}")
            # Record failure entry so the registry stays consistent
            records.append({
                "experiment_id": experiment_id,
                "config": config,
                "artifact_refs": {},
                "error": str(e),
            })

    # Upsert into persistent registry
    _upsert_registry(records, base_path)

    return records


def load_registry(base_path: str | Path = "artifacts") -> list[dict]:
    """Load the persisted experiment registry.

    Returns an empty list if no registry exists yet.
    """
    path = _registry_path(base_path)
    if not path.exists():
        return []
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_artifact_refs(config: dict, base_path: Path) -> dict[str, Artifact]:
    """Re-derive Artifact descriptors from config (always deterministic)."""
    refs = {}
    for key, stage, name in _STAGE_ARTIFACT_MAP:
        try:
            refs[key] = make_stage_artifact(config, base_path, stage, name)
        except Exception:
            pass  # stage config key might be missing for partial configs
    return refs


def _registry_path(base_path: Path | str) -> Path:
    return Path(base_path) / "experiments" / _REGISTRY_FILENAME


def _upsert_registry(new_records: list[dict], base_path: Path) -> None:
    """Load existing registry, upsert new records by experiment_id, and save."""
    registry_path = _registry_path(base_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    existing: dict[str, dict] = {}
    if registry_path.exists():
        with registry_path.open("rb") as f:
            for rec in pickle.load(f):
                existing[rec["experiment_id"]] = rec

    # Upsert
    for rec in new_records:
        existing[rec["experiment_id"]] = rec

    with registry_path.open("wb") as f:
        pickle.dump(list(existing.values()), f)

    print(f"\n📋 Registry saved: {registry_path} ({len(existing)} experiments)")
