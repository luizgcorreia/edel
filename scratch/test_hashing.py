from pathlib import Path
from edel.io.artifact import stage_hash, CANONICAL_STAGE_NAMES

# Base configuration
config_base = {
    "random_seed": 42,
    "processing_mode": "simple",
    "data": {"provider": {"type": "openalex", "topic_id": "T1"}},
    "structured_abstracts": {"n_documents": 100},
    "embedding": {"model": "text-embedding-3-small"},
    "dimensionality_reduction": {"method": "umap"},
    "vector_field": {"grid_size": 20},
    "clustering": {"c1": {"algorithm": "kmeans"}},
    "labeling": {"enabled": True},
    "landscape": {"metric": "cited"}
}

def print_hashes(cfg, name):
    print(f"\n--- {name} ---")
    for s in CANONICAL_STAGE_NAMES:
        print(f"{s[:20]:<20}: {stage_hash(cfg, s)[:8]}")

# Scenario 1: Base hashes
hashes_base = {s: stage_hash(config_base, s) for s in CANONICAL_STAGE_NAMES}
print_hashes(config_base, "Base Configuration")

# Scenario 2: Change a downstream parameter (vector_field)
config_vf_changed = config_base.copy()
config_vf_changed["vector_field"] = {"grid_size": 50}
print_hashes(config_vf_changed, "Changed Vector Field (Downstream)")

hashes_vf = {s: stage_hash(config_vf_changed, s) for s in CANONICAL_STAGE_NAMES}

# Verify separation
assert hashes_base["data_collection"] == hashes_vf["data_collection"], "Upstream changed incorrectly!"
assert hashes_base["dimensionality_reduction"] == hashes_vf["dimensionality_reduction"], "Upstream changed incorrectly!"
assert hashes_base["vector_field"] != hashes_vf["vector_field"], "VF hash didn't change!"
assert hashes_base["clustering"] != hashes_vf["clustering"], "Clustering hash didn't change!"

# Scenario 3: Change a global parameter (random_seed)
config_global_changed = config_base.copy()
config_global_changed["random_seed"] = 99
print_hashes(config_global_changed, "Changed Global Seed")

hashes_global = {s: stage_hash(config_global_changed, s) for s in CANONICAL_STAGE_NAMES}
for s in CANONICAL_STAGE_NAMES:
    assert hashes_base[s] != hashes_global[s], f"Stage {s} failed to change with global seed!"

print("\n✅ All assertions passed! Chained hashing works perfectly.")
