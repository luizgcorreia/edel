import copy
import json
import hashlib
from edel.io.artifact import stage_hash

def main():
    config1 = {
        "random_seed": 42,
        "embedding_mode": "aspects",
        "data": {"provider": {"type": "openalex", "topic_id": "T1", "region": "global"}},
        "structured_abstracts": {"model": "gpt-4o"},
        "embedding": {"model": "text-embedding-3-small"},
        "dimensionality_reduction": {"method": "umap"},
        "vector_field": {"grid_size": 25},
        "clustering": {"domain": {"algorithm": "hdbscan"}},
        "labeling": {"provider": "openai", "model": "gpt-4o-mini"}
    }
    
    config2 = copy.deepcopy(config1)
    config2["labeling"]["provider"] = "gemini"
    config2["labeling"]["model"] = "gemini-3-flash"
    
    h1 = stage_hash(config1, "clustering")
    h2 = stage_hash(config2, "clustering")
    
    print(f"Config 1 (OpenAI) Clustering Hash: {h1[:8]}")
    print(f"Config 2 (Gemini) Clustering Hash: {h2[:8]}")
    
    if h1 == h2:
        print("✅ SUCCESS: Clustering hashes are identical.")
    else:
        print("❌ FAILURE: Clustering hashes differ!")
        
    # Check labeling hash
    l1 = stage_hash(config1, "labeling")
    l2 = stage_hash(config2, "labeling")
    print(f"Config 1 (OpenAI) Labeling Hash: {l1[:8]}")
    print(f"Config 2 (Gemini) Labeling Hash: {l2[:8]}")
    
if __name__ == "__main__":
    main()
