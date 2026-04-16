"""CLI entrypoint to run and persist a full epistemic landscape experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from edel.config.defaults import RUN_CONFIG
from edel.io.artifacts import save_artifacts
from edel.io.dataset import load_dataset
from edel.pipeline.labels import apply_cluster_labels
from edel.pipeline.run import run_pipeline
from edel.viz.contour import make_contour_figure
from edel.viz.field import add_vector_field_annotations
from edel.viz.scatter import make_scatter


def _build_config(dataset_path: str | None, field_dataset_path: str | None) -> dict:
    config = deepcopy(RUN_CONFIG)
    if dataset_path:
        config["data"]["dataset_path"] = dataset_path
    if field_dataset_path:
        config["paths"]["field_dataset_path"] = field_dataset_path
    return config


def main() -> None:
    """Run pipeline, save artifacts, and optionally emit figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Path to the main dataset CSV.")
    parser.add_argument("--field-dataset", default=None, help="Path to vector field CSV.")
    parser.add_argument("--artifacts", default=None, help="Override artifact output path.")
    parser.add_argument("--make-plots", action="store_true", help="Generate HTML plots.")
    args = parser.parse_args()

    config = _build_config(args.dataset, args.field_dataset)
    artifacts = run_pipeline(config)

    artifacts_path = args.artifacts or config["paths"]["artifacts_path"]
    save_path = save_artifacts(artifacts, artifacts_path)
    print(f"Saved artifacts to: {save_path}")

    if not args.make_plots:
        return

    plots_dir = Path(config["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    method = config["dimensionality_reduction"]["method"]
    x_label = f"proj_p_{method}_x"
    y_label = f"proj_p_{method}_y"
    topic = config["data"]["provider"].get("topic_name", "Dataset")

    contour = make_contour_figure(
        xi=artifacts["grid"]["xi"],
        yi=artifacts["grid"]["yi"],
        grid_smooth=artifacts["grid"]["grid_smooth"],
        z_label=artifacts["height"]["label"],
        title=f"2D Epistemic Landscape Map ({topic})",
        x_label=x_label,
        y_label=y_label,
    )
    contour = add_vector_field_annotations(contour, artifacts.get("field_vectors"))
    contour.write_html(plots_dir / "contour_map.html")

    df = artifacts["df"].copy()
    cluster_labels_path = config.get("paths", {}).get("cluster_labels_path")
    cluster_labels_df = load_dataset(cluster_labels_path) if cluster_labels_path and Path(cluster_labels_path).exists() else None

    color_cluster = config["landscape"].get("color_cluster", "cluster_domain")
    style_cluster = config["landscape"].get("style_cluster", "cluster_style")

    color_col = None
    style_col = None

    if color_cluster in df.columns:
        if cluster_labels_df is not None:
            df["_color_label"] = apply_cluster_labels(df, cluster_labels_df, color_cluster)
            color_col = "_color_label"
        else:
            color_col = color_cluster

    if style_cluster in df.columns:
        if cluster_labels_df is not None:
            df["_style_label"] = apply_cluster_labels(df, cluster_labels_df, style_cluster)
            style_col = "_style_label"
        else:
            style_col = style_cluster

    scatter = make_scatter(
        df=df,
        X=artifacts["projection"]["X"],
        Y=artifacts["projection"]["Y"],
        color_col=color_col,
        style_col=style_col,
    )
    scatter.update_layout(title=f"Projection Scatter ({topic})")
    scatter.write_html(plots_dir / "projection_scatter.html")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
