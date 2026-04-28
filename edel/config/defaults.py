"""Default run configuration mirroring the original monolithic pipeline."""

RUN_CONFIG = {
    "processing_mode": "simple",  # "simple" | "batch"
    "random_seed": 42,
    "embedding_mode": "aspects",  # "aspects" | "documents"
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
        "transforms": [{"type": "shuffle_words"}],
    },
    "structured_abstracts": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "min_sentences": 4,
        "min_tokens": 80
    },
    "embedding": {
        "mode": "multi",  # multi | single | abstract
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "n_dimensions": 1536,
    },
    "dimensionality_reduction": {
        "method": "diffusion",
        "n_neighbors": 15,
        "random_state": 0,
        "min_dist": 0.1,
        "metric": "cosine",
    },
    "vector_field": {
        "method": "diffusion",
        "grid_size": 25,
        "min_count": 3,
        "smooth_sigma": 1.0,
        "compute_divergence": True,
        "compute_magnitude": True,
    },
    "clustering": {
        "domain": {
            "source": "proj_p",
            "algorithm": "hdbscan",
            "params": {
                "min_cluster_size": 15,
            },
        },
        "field": {
            "source": "field",
            "algorithm": "hdbscan",
            "params": {
                "min_cluster_size": 10,
            },
        },
        "style": {
            "source": "features",
            "algorithm": "gmm",
            "params": {
                "n_components": 4,
            },
        },
        "operator": {
            "source": "operators",
            "algorithm": "hdbscan",
            "params": {
                "min_cluster_size": 20,
            },
        },
    },
    "labeling": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "text_column": "abstract_text",
        "topic": None,
        "language": "en",
        "axis": {
            "enabled": True,
            "projection": "diffusion",
            "n_samples": 5,
        },
        "clusters": {
            "enabled": True,
            "cluster_keys": ["domain", "style", "operator", "field"],
            "n_samples": 5,
        },
    },
    "landscape": {
        "metric": "cited_by_count",
        "log_scale": True,
        "grid": {
            "num_bins": 50,
            "sigma": 1.5,
        },
        "scale": 1.0,
        "color_cluster": "cluster_domain",
        "style_cluster": "cluster_style",
        "field": {
            "enabled": True,
            "type": "fi",
            "step": 2,
            "scale": 0.20,
            "width": 0.7,
            "arrow_size": 0.8,
        },
    },
}
