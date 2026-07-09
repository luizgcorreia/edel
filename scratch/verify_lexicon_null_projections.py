import json
import pandas as pd
import numpy as np
from edel.pipeline.projection import run_projection_stage

# 1. Load configuration for afp_lexicon_null
with open("/home/correia/edel/artifacts/configs/registry.json", "r") as f:
    registry = json.load(f)

# Find config for afp_lexicon_null
config = registry.get("afp_lexicon_null")
if not config:
    # fallback to some default
    print("Could not find afp_lexicon_null in registry.json, using fallback config")
    config = {
        "embedding": {"n_dimensions": 1536},
        "dimensionality_reduction": {
            "method": "diffusion",
            "random_state": 42,
        }
    }
else:
    print("Found config in registry.json")

# 2. Load embeddings parquet file
embeddings_path = "/home/correia/edel/artifacts/embeddings/lexicon_null_none_global/embeddings_f6fcb4e3.parquet"
df = pd.read_parquet(embeddings_path)
print("Loaded embeddings shape:", df.shape)

# Let's check how many embedding columns are empty/null before projection
aspects = ["problem", "method", "finding", "interpretation"]
print("\nBefore projection - null embeddings:")
for a in aspects:
    col = f"{a}_embedding"
    if col in df.columns:
        null_count = df[col].isna().sum()
        print(f"  {col}: {null_count} / {len(df)}")

# 3. Run projection stage
print("\nRunning projection stage...")
df_proj, report = run_projection_stage(df, config)
print("Finished projection stage.")
print("Report:", report)

# 4. Check projection coordinates
proj_cols = [c for c in df_proj.columns if "proj_" in c]
print("\nProjection columns:")
for col in proj_cols:
    nan_count = df_proj[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs out of {len(df_proj)}")

# 5. Check transition signatures & magnitudes
sig_cols = ["mag_pm", "mag_mf", "mag_fi", "cos_pm_mf", "cos_mf_fi", "cos_pm_fi"]
print("\nSignatures & Magnitudes:")
for col in sig_cols:
    if col in df_proj.columns:
        nan_count = df_proj[col].isna().sum()
        print(f"  {col}: {nan_count} NaNs out of {len(df_proj)}")
