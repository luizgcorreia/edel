"""Deterministic artifact addressing and persistence utilities.

Artifacts are written under:
    <base_path>/<stage>/<config_hash>/<name>.(parquet|pkl)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import hashlib
from typing import Any


CANONICAL_STAGE_NAMES = (
    "raw_data",
    "sa_data",
    "embeddings_data",
    "dr_data",
    "vf_data",
    "clustering_data",
    "labeled_data",
    "output",
)

# Stage-specific keys in RUN_CONFIG (mirrors the original pipeline organization).
STAGE_CONFIG_KEYS = {
    "raw_data": "data",
    "sa_data": "structured_abstracts",
    "embeddings_data": "embedding",
    "dr_data": "dimensionality_reduction",
    "vf_data": "vector_field",
    "clustering_data": "clustering",
    "labeled_data": "labeling",
    "output": "landscape",
}

CANONICAL_ARTIFACT_NAMES = {
    "raw_data": ("raw",),
    "sa_data": ("sa",),
    "embeddings_data": ("embeddings", "embeddings_intermidiate", "embeddings_batch"),
    "dr_data": ("dr",),
    "vf_data": ("vf",),
    "clustering_data": ("clustering", "field_clustering"),
    "labeled_data": ("labeled", "field_labeled", "axes_labeled"),
    "output": ("plot", "experiment_stats"),
}


@dataclass(frozen=True)
class Artifact:
    """Deterministic descriptor for a pipeline artifact location."""

    stage: str
    name: str
    config_hash: str
    base_path: Path

    @property
    def path_prefix(self) -> Path:
        """Path prefix without extension."""
        return canonical_artifact_path(
            base_path=self.base_path,
            stage=self.stage,
            config_hash=self.config_hash,
            name=self.name,
        )

    @property
    def parquet_path(self) -> Path:
        """Parquet file path for DataFrame artifacts."""
        return self.path_prefix.with_suffix(".parquet")

    @property
    def pkl_path(self) -> Path:
        """Pickle file path for generic Python artifacts."""
        return self.path_prefix.with_suffix(".pkl")


def stable_hash(config: dict) -> str:
    """Return a stable MD5 hash from a JSON-serializable config dict."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def canonical_artifact_path(
    base_path: str | Path,
    stage: str,
    config_hash: str,
    name: str,
) -> Path:
    """Build canonical artifact path prefix: <base>/<stage>/<hash>/<name>."""
    return Path(base_path) / stage / config_hash / name


def stage_hash(run_config: dict, stage: str) -> str:
    """Compute deterministic hash for one canonical pipeline stage."""
    key = STAGE_CONFIG_KEYS[stage]
    return stable_hash(run_config[key])


def make_stage_artifact(
    run_config: dict,
    base_path: str | Path,
    stage: str,
    name: str,
) -> Artifact:
    """Create artifact descriptor for a canonical stage using stage-specific hash."""
    return Artifact(
        stage=stage,
        name=name,
        config_hash=stage_hash(run_config, stage),
        base_path=Path(base_path),
    )


def make_artifact(stage: str, name: str, config: dict, base_path: str | Path) -> Artifact:
    """Create an artifact descriptor addressed deterministically by config hash."""
    return Artifact(
        stage=stage,
        name=name,
        config_hash=stable_hash(config),
        base_path=Path(base_path),
    )


def _is_pandas_dataframe(obj: Any) -> bool:
    """Check DataFrame type without requiring a hard pandas import at module import time."""
    return obj.__class__.__name__ == "DataFrame" and obj.__class__.__module__.startswith("pandas")


def save_artifact(artifact: Artifact, obj: Any) -> Path:
    """Persist artifact object as parquet (DataFrame) or pickle (everything else)."""
    artifact.path_prefix.parent.mkdir(parents=True, exist_ok=True)

    if _is_pandas_dataframe(obj):
        path = artifact.parquet_path
        obj.to_parquet(path, index=False)
        return path

    path = artifact.pkl_path
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return path


def load_artifact(artifact: Artifact):
    """Load artifact from deterministic location, preferring parquet over pickle."""
    if artifact.parquet_path.exists():
        import pandas as pd

        return pd.read_parquet(artifact.parquet_path)

    if artifact.pkl_path.exists():
        with artifact.pkl_path.open("rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"Artifact not found at '{artifact.parquet_path}' or '{artifact.pkl_path}'."
    )


if __name__ == "__main__":
    # Small deterministic usage example.
    cfg = {"stage": "example", "version": 1, "params": {"alpha": 0.1}}
    art = make_artifact(stage="output", name="dummy_payload", config=cfg, base_path="artifacts")
    saved_to = save_artifact(art, {"ok": True, "values": [1, 2, 3]})
    loaded = load_artifact(art)

    print(f"Saved to: {saved_to}")
    print(f"Loaded: {loaded}")
