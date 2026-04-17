"""Stage 1 orchestrator: data collection/generation."""

from __future__ import annotations

from pathlib import Path

from edel.io.artifact import make_artifact, save_artifact
from edel.providers import get_provider


def run_data_stage(config: dict, artifact_root: Path) -> dict:
    """Run Stage 1 data collection and save the dataset artifact.

    Args:
        config: Stage-1 config, expected to contain ``config['provider']['type']``.
        artifact_root: Root directory where artifacts are persisted.

    Returns:
        dict[str, Artifact]
    """
    provider_name = config["provider"]["type"]
    provider = get_provider(provider_name)

    dataset_df = provider(config)

    dataset_artifact = make_artifact(
        stage="data_collection",
        name="dataset",
        config=config,
        base_path=artifact_root,
    )
    save_artifact(dataset_artifact, dataset_df)

    return {"dataset": dataset_artifact}


# Example:
# from pathlib import Path
# from edel.config.defaults import RUN_CONFIG
# artifacts = run_data_stage(RUN_CONFIG["data"], Path("artifacts"))
# print(artifacts["dataset"].parquet_path)
