import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import normalize as sk_normalize

from edel.io.artifact import load_artifact, make_stage_artifact
from edel.experiments.registry import get_experiment, init_registry
from edel.experiments.metrics.hypothesis_tests import compute_h2_for_transition
from edel.pipeline.projection import load_embeddings_to_matrix

# Load afp_baseline config and artifacts from registry.pkl
import pickle
registry_path = Path("artifacts/experiments/registry.pkl")
with registry_path.open("rb") as f:
    records = pickle.load(f)
baseline_record = next(rec for rec in records if rec["experiment_id"] == "afp_baseline")
config = baseline_record["config"]
emb_art = baseline_record["artifact_refs"]["embedding"]
df = load_artifact(emb_art)

# Load and normalize matrices
dimensions = config.get("embedding", {}).get("n_dimensions", 1536)
aspects = ["problem", "method", "finding", "interpretation"]

def load(aspect: str) -> np.ndarray:
    mat = load_embeddings_to_matrix(df, f"{aspect}_embedding", dimensions)
    mat -= mat.mean(axis=0)
    return sk_normalize(mat)

emb = {
    "p": load("problem"),
    "m": load("method"),
    "f": load("finding"),
    "i": load("interpretation"),
}

combinations = [
    ("pm", "p", "m", "D(M|p)"),
    ("pf", "p", "f", "D(F|p)"),
    ("pi", "p", "i", "D(I|p)"),
    ("mp", "m", "p", "D(P|m)"),
    ("mf", "m", "f", "D(F|m)"),
    ("mi", "m", "i", "D(I|m)"),
    ("fp", "f", "p", "D(P|f)"),
    ("fm", "f", "m", "D(M|f)"),
    ("fi", "f", "i", "D(I|f)"),
    ("ip", "i", "p", "D(P|i)"),
    ("im", "i", "m", "D(M|i)"),
    ("if", "i", "f", "D(F|i)"),
]

print("Calculating all 12 combinations for afp_baseline...")
results = []
np.random.seed(42)
for key, x_name, y_name, label in combinations:
    w_obs, p_val = compute_h2_for_transition(emb[x_name], emb[y_name], B=20)
    results.append({
        "Key": key,
        "Label": label,
        "W_obs": w_obs,
        "p-value": p_val
    })

res_df = pd.DataFrame(results)
print("\nResults:")
print(res_df.to_string(index=False))
print(f"\nNumber of significant transitions (p < 0.05): {sum(res_df['p-value'] < 0.05)}")
