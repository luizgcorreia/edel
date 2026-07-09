import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from edel.experiments.runner import load_registry
from edel.dashboard.cache import rebuild_results_cache

print("--- Initial load_registry ---")
reg = load_registry("artifacts")
print("Experiments in registry.pkl:", [x["experiment_id"] for x in reg])

print("\n--- Rebuilding results cache ---")
df = rebuild_results_cache("artifacts", delta_only=False)
print("Experiment IDs in results cache:", df["experiment_id"].tolist() if not df.empty else [])
