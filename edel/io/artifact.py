"""Deterministic artifact addressing and persistence utilities.

Artifacts are written under:
    <base_path>/<stage>/<label>/<name>_<hash_segment>.(parquet|pkl)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import hashlib
from typing import Any


CANONICAL_STAGE_NAMES = (
    "data_collection",
    "structured_abstracts",
    "embeddings",
    "dimensionality_reduction",
    "vector_field",
    "clustering",
    "labeling",
    "output",
)

# Parameters that trigger a re-hash for all stages if changed
GLOBAL_CONFIG_KEYS = ("random_seed", "embedding_mode")

# Stage-specific keys in RUN_CONFIG (mirrors the original pipeline organization).
STAGE_CONFIG_KEYS = {
    "data_collection": "data",
    "structured_abstracts": "structured_abstracts",
    "embeddings": "embedding",
    "dimensionality_reduction": "dimensionality_reduction",
    "vector_field": "vector_field",
    "clustering": "clustering",
    "labeling": "labeling",
    "output": "landscape",
}

CANONICAL_ARTIFACT_NAMES = {
    "data_collection": ("dataset", "filter_report", "raw"),
    "structured_abstracts": ("sa", "filter_report"),
    "embeddings": ("embeddings", "embeddings_intermidiate", "embeddings_batch"),
    "dimensionality_reduction": ("dr", "report"),
    "vector_field": ("vf",),
    "clustering": ("clustering", "field_clustering"),
    "labeling": ("labeled", "field_labeled", "axes_labeled"),
    "output": ("landscape_results", "experiment_stats"),
}


def get_experiment_label(config: dict) -> str:
    """Derive experiment label (provider_topicId_region) from config.

    Supports both full RUN_CONFIG and its 'data' section.
    """
    data_cfg = config.get("data", config)
    provider_cfg = data_cfg.get("provider", {})

    provider = provider_cfg.get("type", "unknown")
    topic_id = provider_cfg.get("topic_id", "unknown")

    # Try region_label, then region, then default to 'global'
    region = provider_cfg.get("region_label") or provider_cfg.get("region") or "global"

    return f"{provider}_{topic_id}_{region}"


@dataclass(frozen=True)
class Artifact:
    """Deterministic descriptor for a pipeline artifact location."""

    stage: str
    name: str
    label: str
    config_hash: str
    base_path: Path

    @property
    def path_prefix(self) -> Path:
        """Path prefix without extension."""
        return canonical_artifact_path(
            base_path=self.base_path,
            stage=self.stage,
            label=self.label,
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
    label: str,
    config_hash: str,
    name: str,
) -> Path:
    """Build canonical artifact path: <base>/<stage>/<label>/<name>_<hash_segment>."""
    hash_segment = config_hash[:8]
    return Path(base_path) / stage / label / f"{name}_{hash_segment}"


def get_global_hash(run_config: dict) -> str:
    """Compute a stable hash for parameters that affect all stages."""
    global_cfg = {k: run_config.get(k) for k in GLOBAL_CONFIG_KEYS if k in run_config}
    return stable_hash(global_cfg)


def stage_hash(run_config: dict, stage: str) -> str:
    """Compute deterministic chained hash for a pipeline stage.
    
    The hash is a linked list: Hash_N = MD5(Hash_{N-1} + Stage_N_Params).
    """
    if stage not in CANONICAL_STAGE_NAMES:
        return stable_hash(run_config.get(STAGE_CONFIG_KEYS.get(stage, stage), {}))
        
    current_hash = get_global_hash(run_config)
    
    for s in CANONICAL_STAGE_NAMES:
        stage_key = STAGE_CONFIG_KEYS[s]
        stage_params = run_config.get(stage_key, {})
        stage_params_hash = stable_hash(stage_params)
        
        # Chain the previous hash with the current stage params hash
        current_hash = stable_hash({"prev_hash": current_hash, "params_hash": stage_params_hash})
        
        if s == stage:
            return current_hash
            
    raise ValueError(f"Unknown stage: {stage}")


def make_stage_artifact(
    run_config: dict,
    base_path: str | Path,
    stage: str,
    name: str,
) -> Artifact:
    """Create artifact descriptor for a canonical stage using chained stage-specific hash."""
    label = get_experiment_label(run_config)
    return Artifact(
        stage=stage,
        name=name,
        label=label,
        config_hash=stage_hash(run_config, stage),
        base_path=Path(base_path),
    )


def make_artifact(
    stage: str,
    name: str,
    config: dict,
    base_path: str | Path,
    run_config: dict | None = None,
    label: str | None = None,
) -> Artifact:
    """Create an artifact descriptor addressed deterministically by config hash."""
    if not label:
        # Try to get label from run_config or config (if it's the data section)
        label = get_experiment_label(run_config or config)

    return Artifact(
        stage=stage,
        name=name,
        label=label,
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


def load_artifact(artifact: Artifact) -> Any:
    """Load artifact data from Parquet (preferred) or Pickle."""
    if artifact.parquet_path.exists():
        import pandas as pd
        return pd.read_parquet(artifact.parquet_path)
    if artifact.pkl_path.exists():
        with open(artifact.pkl_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"Artifact not found at '{artifact.parquet_path}' or '{artifact.pkl_path}'."
    )


def save_viz(artifact: Artifact, fig: Any, formats: list[str] = ["png", "html"]):
    """Save a visualization (Matplotlib or Plotly) using artifact conventions."""
    for fmt in formats:
        hash_segment = artifact.config_hash[:8]
        path = artifact.path_prefix.parent / f"{artifact.name}_{hash_segment}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Plotly support
        if hasattr(fig, "write_html"):
            if fmt == "html":
                fig.write_html(str(path))
            elif fmt == "png":
                try:
                    # Scale=2 for high-res publication quality
                    fig.write_image(str(path), scale=2)
                except Exception as e:
                    print(f"Warning: Could not save Plotly PNG (is 'kaleido' installed?): {e}")
        
        # Matplotlib support
        elif hasattr(fig, "savefig"):
            if fmt == "png":
                fig.savefig(str(path), dpi=300, bbox_inches="tight")
            # PDF is often preferred for papers
            elif fmt == "pdf":
                fig.savefig(str(path), bbox_inches="tight")
        
        print(f"✅ Saved visualization: {path.name}")


if __name__ == "__main__":
    # Small deterministic usage example mirroring RUN_CONFIG structure.
    run_cfg = {
        "data": {
            "provider": {
                "type": "openalex",
                "topic_id": "T10102",
                "region": None,
            }
        },
        "landscape": {"metric": "cited_by_count"},
    }
    art = make_artifact(
        stage="output",
        name="dummy_payload",
        config=run_cfg["landscape"],
        run_config=run_cfg,
        base_path="artifacts",
    )
    saved_to = save_artifact(art, {"ok": True, "values": [1, 2, 3]})
    loaded = load_artifact(art)

    print(f"Saved to: {saved_to}")
    print(f"Loaded: {loaded}")
def delete_experiment_artifacts(config: dict, base_path: str | Path):
    """Delete all files associated with the given config across all stages.
    
    This computes the hash for each stage and deletes any file containing that hash segment.
    """
    base_path = Path(base_path)
    label = get_experiment_label(config)
    
    deleted_count = 0
    for stage in CANONICAL_STAGE_NAMES:
        try:
            h = stage_hash(config, stage)[:8]
            stage_dir = base_path / stage / label
            if not stage_dir.exists():
                continue
                
            # Find and delete files with this hash segment
            for f in stage_dir.glob(f"*_{h}.*"):
                try:
                    f.unlink()
                    print(f"🗑️ Deleted artifact: {f.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
        except KeyError:
            continue # Stage key might be missing in config
            
    return deleted_count
