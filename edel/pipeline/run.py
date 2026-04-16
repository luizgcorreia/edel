"""Single-entry orchestration for the full epistemic landscape pipeline."""

from edel.io.dataset import load_dataset
from edel.pipeline.field import add_vector_field_2d_smooth
from edel.pipeline.grid import make_height_grid, smooth_grid
from edel.pipeline.height import compute_height_metric
from edel.pipeline.projection import get_projection_xy


def run_pipeline(config: dict) -> dict:
    """Run the full pipeline and return reusable artifacts."""
    df = load_dataset(config["data"]["dataset_path"])

    method = config["dimensionality_reduction"]["method"]
    scale = config["landscape"].get("scale", 8.0)
    X, Y = get_projection_xy(df, method=method, scale=scale)

    metric = config["landscape"].get("metric", "cited_by_count")
    log_scale = config["landscape"].get("log_scale", True)
    Z, z_label = compute_height_metric(df, metric=metric, log_scale=log_scale)

    grid_cfg = config["landscape"].get("grid", {})
    num_bins = grid_cfg.get("num_bins", 40)
    sigma = grid_cfg.get("sigma", 2.0)

    xi, yi, grid = make_height_grid(X, Y, Z, num_bins=num_bins)
    grid_smooth = smooth_grid(grid, sigma=sigma)

    field_df = None
    field_vectors = None
    field_dataset_path = config.get("paths", {}).get("field_dataset_path")
    if field_dataset_path:
        field_df = load_dataset(field_dataset_path)
        field_cfg = config["landscape"].get("field", {})
        field_vectors = add_vector_field_2d_smooth(
            field_df=field_df,
            field_type=field_cfg.get("field_type", "discovery"),
            xy_scale=scale,
            grid_res=field_cfg.get("grid_res", 40),
            kernel_sigma=field_cfg.get("kernel_sigma", 0.25),
            step=field_cfg.get("step", 2),
            scale=field_cfg.get("scale", 0.14),
        )

    return {
        "df": df,
        "projection": {"X": X, "Y": Y},
        "height": {"Z": Z, "label": z_label},
        "grid": {
            "xi": xi,
            "yi": yi,
            "grid": grid,
            "grid_smooth": grid_smooth,
        },
        "field": field_df,
        "field_vectors": field_vectors,
    }
