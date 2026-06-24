"""Report generator — consolidated Excel report for hypothesis tests.

Usage:
    python -m edel.experiments.report \\
        --hypotheses H1 \\
        --experiments afp_baseline afp_lexicon_null \\
        --output report.xlsx \\
        --force
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from edel.experiments.runner import load_registry
from edel.experiments.analyzer import analyze_experiments

logger = logging.getLogger(__name__)

# Metadata columns to include in every tab
META_COLUMNS = [
    "experiment_id",
    "provider",
    "topic_name",
    "n_documents_requested",
    "embedding_model",
    "projection_method",
]

# 12 directional transitions
_TRANSITIONS = ["pm", "pf", "pi", "mp", "mf", "mi", "fp", "fm", "fi", "ip", "im", "if"]
_PAIRS = ["pm", "mf", "fi", "pf", "pi", "mi"]

HYPOTHESIS_COLUMNS: dict[str, dict[str, list[str]]] = {
    "H1": {
        "Primary": ["h1_energy_stat", "h1_energy_pvalue"],
        "Wasserstein Effect Sizes": ["h1_w_norm_pm", "h1_w_norm_mf", "h1_w_norm_fi", "h1_w_cos_pm_mf", "h1_w_cos_pm_fi", "h1_w_cos_mf_fi"],
        "KS (stat)": ["h1_ks_stat_norm_pm", "h1_ks_stat_norm_mf", "h1_ks_stat_norm_fi", "h1_ks_stat_cos_pm_mf", "h1_ks_stat_cos_pm_fi", "h1_ks_stat_cos_mf_fi"],
        "KS (pvalue)": ["h1_ks_pvalue_norm_pm", "h1_ks_pvalue_norm_mf", "h1_ks_pvalue_norm_fi", "h1_ks_pvalue_cos_pm_mf", "h1_ks_pvalue_cos_pm_fi", "h1_ks_pvalue_cos_mf_fi"],
    },
    "H2": {
        "Primary (p-values)": [f"h2_pvalue_{t}" for t in _TRANSITIONS],
        "Wasserstein Distance": [f"h2_w_dist_{t}" for t in _TRANSITIONS],
        "Z-scores": [f"h2_z_{t}" for t in _TRANSITIONS],
        "Asymmetry (diff)": [f"h2b_diff_{p}" for p in _PAIRS],
        "Asymmetry (p-value)": [f"h2b_pvalue_{p}" for p in _PAIRS],
    },
    "H3": {
        "Primary": ["h3_w_edel", "h3_w_baseline", "h3_predictive_gain", "h3_gain_pvalue", "h3_moran_i", "h3_moran_pvalue"],
    },
}


def _all_hypothesis_columns(hypothesis: str) -> list[str]:
    groups = HYPOTHESIS_COLUMNS.get(hypothesis, {})
    return [c for cols in groups.values() for c in cols]


def generate_report(
    experiment_ids: list[str] | None = None,
    hypotheses: list[str] | None = None,
    base_path: str | Path = "artifacts",
    force: bool = False,
    output_path: str | Path = "report.xlsx",
) -> Path:
    """Generate consolidated Excel report for selected hypotheses and experiments."""
    base_path = Path(base_path)
    output_path = Path(output_path)

    if hypotheses is None:
        hypotheses = list(HYPOTHESIS_COLUMNS.keys())

    # ── Get results DataFrame ────────────────────────────────────────────
    if force:
        registry = load_registry(base_path)
        if experiment_ids:
            id_set = set(experiment_ids)
            records = [r for r in registry if r["experiment_id"] in id_set]
            missing = id_set - {r["experiment_id"] for r in records}
            if missing:
                logger.warning(f"Experiments not in registry: {missing}")
        else:
            records = registry
            experiment_ids = [r["experiment_id"] for r in records]

        if not records:
            logger.error("No matching experiments found in registry.")
            return output_path

        df = analyze_experiments(records, base_path=base_path)
    else:
        cache_path = base_path / "experiments" / "results.parquet"
        if not cache_path.exists():
            logger.error(f"Cache not found at {cache_path}. Use --force to recompute.")
            return output_path
        df = pd.read_parquet(cache_path)
        if experiment_ids:
            df = df[df["experiment_id"].isin(experiment_ids)]

    if df.empty:
        logger.error("No results to report.")
        return output_path

    # ── Build Excel ──────────────────────────────────────────────────────
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for ht in hypotheses:
            if ht not in HYPOTHESIS_COLUMNS:
                logger.warning(f"Unknown hypothesis: {ht}, skipping.")
                continue

            all_cols = _all_hypothesis_columns(ht)
            available = [c for c in all_cols if c in df.columns]
            meta_available = [c for c in META_COLUMNS if c in df.columns]

            if not available:
                logger.warning(f"No columns found for {ht}, skipping tab.")
                continue

            tab_cols = meta_available + [c for c in available if c not in set(meta_available)]
            tab_df = df[tab_cols].copy()

            # Add pass/fail columns
            if ht == "H1" and "h1_energy_pvalue" in tab_df.columns:
                tab_df["h1_pass"] = tab_df["h1_energy_pvalue"] < 0.05
            elif ht == "H2":
                pval_cols = [c for c in tab_df.columns if c.startswith("h2_pvalue_") and c != "h2_pvalue"]
                if pval_cols:
                    tab_df["h2_num_passed"] = (tab_df[pval_cols] < 0.05).sum(axis=1)
                    tab_df["h2_pass"] = tab_df["h2_num_passed"] >= 6
            elif ht == "H3" and "h3_predictive_gain" in tab_df.columns and "h3_gain_pvalue" in tab_df.columns:
                tab_df["h3_pass"] = (tab_df["h3_predictive_gain"] > 0) & (tab_df["h3_gain_pvalue"] < 0.05)

            tab_df.to_excel(writer, sheet_name=ht, index=False)
            logger.info(f"  Tab '{ht}': {len(tab_df)} rows × {len(tab_cols)} columns")

    logger.info(f"Report saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated HT report")
    parser.add_argument("--hypotheses", nargs="+", choices=["H1", "H2", "H3", "all"],
                        default=["all"], help="Hypotheses to include")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Experiment IDs (default: all in registry)")
    parser.add_argument("--output", default="report.xlsx",
                        help="Output Excel path")
    parser.add_argument("--force", action="store_true",
                        help="Recompute metrics from artifacts (ignore cache)")
    parser.add_argument("--base-path", default="artifacts",
                        help="Artifact base path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    hypotheses = list(HYPOTHESIS_COLUMNS.keys()) if "all" in args.hypotheses else args.hypotheses
    generate_report(
        experiment_ids=args.experiments,
        hypotheses=hypotheses,
        base_path=args.base_path,
        force=args.force,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
