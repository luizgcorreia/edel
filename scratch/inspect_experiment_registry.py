import pickle
from pathlib import Path
import pandas as pd
import numpy as np

registry_path = Path("/home/correia/edel/artifacts/experiments/registry.pkl")
with registry_path.open("rb") as f:
    records = pickle.load(f)

for rec in records:
    eid = rec["experiment_id"]
    if "lexicon_null" in eid:
        print(f"\n=================== {eid} ===================")
        # print(f"Config: {rec['config']}")
        print("Artifact refs:")
        for k, art in rec["artifact_refs"].items():
            print(f"  {k}: parquet={art.parquet_path}, exists={art.parquet_path.exists()}")
            
            # Let's inspect the dimensionality reduction parquet file!
            if k == "projection" and art.parquet_path.exists():
                df = pd.read_parquet(art.parquet_path)
                print(f"    DR Shape: {df.shape}")
                
                # Check for nulls in coordinates
                proj_cols = [c for c in df.columns if "proj_" in c and ("_x" in c or "_y" in c)]
                print(f"    Projection Columns: {proj_cols}")
                for col in proj_cols:
                    nulls = df[col].isnull().sum()
                    print(f"      {col}: nulls={nulls}/{len(df)}")
                    # Sample values
                    non_nulls = df[col].dropna()
                    if len(non_nulls) > 0:
                        print(f"        Sample values: {non_nulls.head(3).tolist()}")
                        print(f"        Range: min={non_nulls.min():.6f}, max={non_nulls.max():.6f}, std={non_nulls.std():.6f}")

                # Let's check aspect embeddings
                aspect_cols = [c for c in df.columns if "embedding" in c]
                print(f"    Embedding columns found: {aspect_cols}")
                for col in aspect_cols:
                    nulls = df[col].isnull().sum()
                    print(f"      {col}: nulls={nulls}/{len(df)}")
