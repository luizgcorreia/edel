from pathlib import Path
import pickle
import pandas as pd

base_path = Path("artifacts")
registry_path = base_path / "experiments" / "registry.pkl"
cache_path = base_path / "experiments" / "results.parquet"

# 1. Clean registry.pkl
if registry_path.exists():
    with registry_path.open("rb") as f:
        records = pickle.load(f)
    cleaned_records = [r for r in records if r["experiment_id"] != "openalex_T10102_global"]
    print(f"Registry: {len(records)} -> {len(cleaned_records)} records")
    with registry_path.open("wb") as f:
        pickle.dump(cleaned_records, f)
else:
    print("registry.pkl not found")

# 2. Clean results.parquet
if cache_path.exists():
    df = pd.read_parquet(cache_path)
    cleaned_df = df[df["experiment_id"] != "openalex_T10102_global"]
    print(f"Cache DataFrame: {len(df)} -> {len(cleaned_df)} rows")
    cleaned_df.to_parquet(cache_path, index=False)
else:
    print("results.parquet not found")
