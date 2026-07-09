import copy
from edel.io.artifact import stage_hash

def main():
    # Base config with OpenAI defaults (likely the original state)
    base_config = {
        "random_seed": 42,
        "embedding_mode": "aspects",
        "data": {
            "provider": {
                "type": "openalex",
                "topic_id": "T10102",
                "topic_name": "Scientometrics",
                "region": None,
                "params": {"n_documents": 300, "avg_length": 150},
            },
            "transforms": [],
        },
        "structured_abstracts": {"provider": "openai", "model": "gpt-4o-mini", "min_sentences": 4, "min_tokens": 80},
        "embedding": {"mode": "multi", "provider": "openai", "model": "text-embedding-ada-002", "n_dimensions": 1536},
        "dimensionality_reduction": {"method": "diffusion", "n_neighbors": 15, "random_state": 0, "min_dist": 0.1, "metric": "cosine"},
        "vector_field": {"method": "diffusion", "grid_size": 25, "min_count": 3, "smooth_sigma": 1.0, "compute_divergence": True, "compute_magnitude": True},
        "clustering": {
            "domain": {"source": "proj_p", "algorithm": "hdbscan", "params": {"min_cluster_size": 15}},
            "field": {"source": "field", "algorithm": "hdbscan", "params": {"min_cluster_size": 10}},
            "style": {"source": "features", "algorithm": "gmm", "params": {"n_components": 4}},
            "operator": {"source": "operators", "algorithm": "hdbscan", "params": {"min_cluster_size": 20}}
        }
    }
    
    # Try different document counts
    for n in [300, 3000, 9000]:
        cfg = copy.deepcopy(base_config)
        cfg["data"]["provider"]["params"]["n_documents"] = n
        h = stage_hash(cfg, "clustering")
        print(f"[n={n}] Clustering Hash: {h[:8]}")
        
    # Try different random seeds
    for s in [0, 1, 42]:
        cfg = copy.deepcopy(base_config)
        cfg["random_seed"] = s
        h = stage_hash(cfg, "clustering")
        print(f"[seed={s}] Clustering Hash: {h[:8]}")

if __name__ == "__main__":
    main()
