import pickle
from pathlib import Path

registry_path = Path("artifacts/experiments/registry.pkl")
with registry_path.open("rb") as f:
    records = pickle.load(f)

for rec in records:
    eid = rec["experiment_id"]
    if eid == "afp_baseline":
        print(f"\n--- config for {eid} ---")
        import pprint
        pprint.pprint(rec["config"])
        print("\n--- artifact_refs ---")
        for k, art in rec["artifact_refs"].items():
            print(f"{k}: parquet={art.parquet_path}, exists={art.parquet_path.exists()}")
