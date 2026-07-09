import json
import pandas as pd
from edel.pipeline.data import run_data_stage
from edel.io.artifact import make_stage_artifact, save_artifact

# 1. Load config
with open("artifacts/configs/registry.json") as f:
    registry = json.load(f)

config = registry["afp_baseline"]

# Ensure we run data collection synchronously and overwrite any existing cached file
print("Running data stage for afp_baseline...")
df, report = run_data_stage(config)

print(f"Generated dataset shape: {df.shape}")

# Save the updated artifacts to disk
art_data = make_stage_artifact(config, "artifacts", "data_collection", "dataset")
art_report = make_stage_artifact(config, "artifacts", "data_collection", "filter_report")

data_path = save_artifact(art_data, df)
report_path = save_artifact(art_report, report)
print(f"Saved dataset artifact to: {data_path}")
print(f"Saved filter report artifact to: {report_path}")

# Print top 15 entries by citation count
top_cited = df.sort_values(by="cited_by_count", ascending=False).head(15)
print("\nTop 15 Cited Entries in regenerated dataset:")
for idx, row in top_cited.iterrows():
    print(f"- {row['id']}: {row['cited_by_count']} citations (Title: {row['title']})")

print("\nOfficial Top 10 Reference:")
official = {
    "List-Index": 26,
    "Show": 21,
    "Collections": 18,
    "Jordan_Normal_Form": 17,
    "Coinductive": 16,
    "Regular-Sets": 15,
    "Polynomial_Factorization": 15,
    "Deriving": 15,
    "Polynomial_Interpolation": 14,
    "Refine_Imperative_HOL": 13
}
for k, v in official.items():
    actual_row = df[df["id"] == k]
    if not actual_row.empty:
        actual_count = actual_row.iloc[0]["cited_by_count"]
        status = "✅ MATCH" if actual_count == v else f"❌ MISMATCH (actual: {actual_count})"
        print(f"  {k}: official={v}, actual={actual_count} -> {status}")
    else:
        print(f"  {k}: NOT FOUND IN DATASET")
