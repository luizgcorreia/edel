"""Default configuration for the epistemic landscape pipeline."""

RUN_CONFIG = {
    "processing_mode": "batch",
    "embedding_mode": "aspects",
    "data": {
        "provider": {
            "type": "openalex",
            "topic_id": "T10102",
            "topic_name": "Scientometrics",
            "region": None,
            "params": {
                "n_documents": 300,
                "avg_length": 150,
            },
        },
        "dataset_path": "data/cluster_data.csv",
        "transforms": [{"type": "shuffle_words"}],
    },
    "dimensionality_reduction": {
        "method": "diffusion",
    },
    "landscape": {
        "scale": 8.0,
        "metric": "cited_by_count",
        "log_scale": True,
        "color_cluster": "cluster_domain",
        "style_cluster": "cluster_style",
        "grid": {
            "num_bins": 40,
            "sigma": 2.0,
        },
        "field": {
            "field_type": "discovery",
            "grid_res": 40,
            "kernel_sigma": 0.25,
            "step": 2,
            "scale": 0.14,
        },
    },
    "paths": {
        "field_dataset_path": "data/field_cluster_data.csv",
        "axis_labels_path": "data/axes_labels.csv",
        "cluster_labels_path": "data/clusters_labels.csv",
        "artifacts_path": "artifacts/pipeline_artifacts.pkl",
        "plots_dir": "artifacts/plots",
    },
}
