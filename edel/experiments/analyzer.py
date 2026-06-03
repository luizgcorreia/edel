"""Analyzer — metrics engine.

Phase 2 of the experiments engine: evidence extraction.

    experiment records → load artifacts → compute_all_metrics → experiment DataFrame

Also provides compare_experiments() for pairwise statistical comparison
(KS-2samp tests) across the feature distributions.
"""

from __future__ import annotations

import logging
import pickle
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from edel.io.artifact import Artifact, load_artifact, make_stage_artifact, save_artifact
from edel.experiments.metrics import METRIC_REGISTRY

logger = logging.getLogger(__name__)

# Feature artifact filename
_FEATURES_FILENAME = "features.pkl"

# Config keys to extract as columns in the result row
_CONFIG_COLUMNS = [
    ("data.provider.type",                 "provider"),
    ("data.provider.topic_id",             "topic_id"),
    ("data.provider.topic_name",           "topic_name"),
    ("data.provider.params.n_documents",   "n_documents_requested"),
    ("embedding.model",                    "embedding_model"),
    ("embedding.n_dimensions",             "embedding_dimensions"),
    ("dimensionality_reduction.method",    "projection_method"),
    ("structured_abstracts.model",         "llm_structuring"),
    ("labeling.model",                     "llm_labeling"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_experiments(
    experiments: list[dict],
    base_path: str | Path = "artifacts",
    save_features: bool = True,
) -> pd.DataFrame:
    """Compute metrics for a list of experiment records.

    For each experiment:
        1. Load pipeline artifacts via artifact_refs.
        2. Run all metric functions in METRIC_REGISTRY (shared context pattern).
        3. Flatten metrics + config keys into one row.
        4. Optionally save per-experiment features for cheap re-analysis.

    Args:
        experiments: List of records from run_experiments().
        base_path: Root artifact directory.
        save_features: If True, save feature distributions as a separate artifact.

    Returns:
        DataFrame where each row is one experiment (the paper dataset).
    """
    rows = []

    for exp in experiments:
        experiment_id = exp["experiment_id"]
        config = exp.get("config", {})
        artifact_refs: dict[str, Artifact] = exp.get("artifact_refs", {})

        if "error" in exp:
            logger.warning(f"Skipping failed experiment: {experiment_id}")
            continue

        logger.info(f"Analyzing: {experiment_id}")
        print(f"\n── Analyzing: {experiment_id} ──")

        try:
            # Build shared context: load all artifacts + pass config metadata
            artifacts = _load_artifacts(artifact_refs)
            artifacts["_dimensions"] = _get(config, "embedding.n_dimensions", 1536)

            # Run all metric functions via registry (shared mutable context)
            all_metrics, all_features = _compute_all_metrics(artifacts)

            # Optionally persist features for cheap re-analysis
            if save_features:
                _save_features(experiment_id, all_features, base_path)

            # Build flat result row
            row = {"experiment_id": experiment_id}
            row.update(_extract_config_columns(config))
            row.update(all_metrics)
            rows.append(row)

            print(f"  ✅ {len(all_metrics)} metrics computed.")

        except Exception as e:
            logger.error(f"Analysis failed for '{experiment_id}': {e}", exc_info=True)
            print(f"  ❌ Failed: {e}")

    return pd.DataFrame(rows)


def compare_experiments(
    experiments: list[dict],
    base_path: str | Path = "artifacts",
    feature_dims: list[str] | None = None,
) -> pd.DataFrame:
    """Pairwise KS-2samp tests across experiments on transition feature distributions.

    Loads saved feature artifacts if available; otherwise re-derives from artifacts.

    Args:
        experiments: List of records from run_experiments().
        base_path: Root artifact directory.
        feature_dims: Feature keys to test. Defaults to the 6 transition features.

    Returns:
        Long-form DataFrame: [exp_a, exp_b, feature, ks_stat, ks_pvalue].
    """
    if feature_dims is None:
        feature_dims = [
            "cos_pm_mf_dist", "cos_pm_fi_dist", "cos_mf_fi_dist",
            "norm_pm_dist",   "norm_mf_dist",   "norm_fi_dist",
        ]

    base_path = Path(base_path)

    # Load feature dicts for each experiment
    feature_store: dict[str, dict] = {}
    for exp in experiments:
        eid = exp["experiment_id"]
        features = _load_saved_features(eid, base_path)
        if features is not None:
            feature_store[eid] = features
        else:
            logger.warning(f"No saved features for '{eid}' — skipping in comparison.")

    if len(feature_store) < 2:
        print("Not enough experiments with features for comparison.")
        return pd.DataFrame()

    rows = []
    for exp_a, exp_b in combinations(feature_store.keys(), 2):
        fa = feature_store[exp_a]
        fb = feature_store[exp_b]

        for dim in feature_dims:
            if dim not in fa or dim not in fb:
                continue

            dist_a = np.asarray(fa[dim]).ravel()
            dist_b = np.asarray(fb[dim]).ravel()

            result = ks_2samp(dist_a, dist_b)
            rows.append({
                "exp_a":     exp_a,
                "exp_b":     exp_b,
                "feature":   dim,
                "ks_stat":   float(result.statistic),
                "ks_pvalue": float(result.pvalue),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_all_metrics(artifacts: dict) -> tuple[dict, dict]:
    """Dispatch all metric functions via METRIC_REGISTRY using shared context."""
    all_metrics: dict = {}
    all_features: dict = {}

    for fn in METRIC_REGISTRY:
        result = fn(artifacts)  # artifacts is the shared mutable context
        all_metrics.update(result.get("metrics", {}))
        all_features.update(result.get("features", {}))

    return all_metrics, all_features


def _load_artifacts(artifact_refs: dict[str, Artifact]) -> dict:
    """Load all artifact objects into a context dict."""
    context: dict[str, Any] = {}
    for key, artifact in artifact_refs.items():
        try:
            val = load_artifact(artifact)
            if isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], pd.DataFrame):
                val = val[0]
            context[key] = val
        except FileNotFoundError:
            logger.debug(f"Artifact '{key}' not found — skipping.")
    return context


def _extract_config_columns(config: dict) -> dict:
    """Extract flat config columns from a nested config dict."""
    row = {}
    for path, col_name in _CONFIG_COLUMNS:
        row[col_name] = _get(config, path, default=None)
    return row


def _get(d: dict, path: str, default: Any = None) -> Any:
    """Safely traverse a dotted path in a nested dict."""
    keys = path.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _features_path(experiment_id: str, base_path: Path) -> Path:
    return base_path / "experiments" / experiment_id / _FEATURES_FILENAME


def _save_features(experiment_id: str, features: dict, base_path: Path | str) -> None:
    path = _features_path(experiment_id, Path(base_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(features, f)
    print(f"  💾 Features saved: {path}")


def _load_saved_features(experiment_id: str, base_path: Path) -> dict | None:
    path = _features_path(experiment_id, base_path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)
