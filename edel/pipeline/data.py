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
    if "data" in config:
        root_cfg = config
        data_cfg = config["data"]
    else:
        root_cfg = {}
        data_cfg = config

    global_seed = root_cfg.get("random_seed")

    provider_name = data_cfg["provider"]["type"]
    provider = get_provider(provider_name)

    import copy
    data_cfg_copy = copy.deepcopy(data_cfg)

    # Inject global seed if not explicitly overridden locally
    if global_seed is not None:
        if "provider" in data_cfg_copy:
            prov = data_cfg_copy["provider"]
            if "params" not in prov:
                prov["params"] = {}
            if "seed" not in prov["params"]:
                prov["params"]["seed"] = global_seed

    dataset_df, report = provider(data_cfg_copy)

    return dataset_df, report
