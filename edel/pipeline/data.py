"""Stage 1 orchestrator: data collection/generation."""

from __future__ import annotations

import pandas as pd
from edel.providers import get_provider


def run_data_stage(config: dict) -> tuple[pd.DataFrame, dict]:
    """Run Stage 1 data collection and return (df, report).

    Args:
        config: Full RUN_CONFIG or the 'data' section.

    Returns:
        tuple[pd.DataFrame, dict]: The collected dataset and a filtering report.
    """
    data_cfg = config.get("data", config)
    provider_name = data_cfg["provider"]["type"]
    provider = get_provider(provider_name)

    dataset_df, report = provider(data_cfg)

    return dataset_df, report
