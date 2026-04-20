"""Stage 1 orchestrator: data collection/generation."""

from __future__ import annotations

import pandas as pd
from edel.providers import get_provider


def run_data_stage(config: dict) -> pd.DataFrame:
    """Run Stage 1 data collection and return the DataFrame.

    Args:
        config: Full RUN_CONFIG or the 'data' section.

    Returns:
        pd.DataFrame: The collected/generated dataset.
    """
    data_cfg = config.get("data", config)
    provider_name = data_cfg["provider"]["type"]
    provider = get_provider(provider_name)

    dataset_df = provider(data_cfg)

    return dataset_df
