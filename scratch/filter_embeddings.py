import json
import shutil
import sys
from pathlib import Path
import pandas as pd

# Resolve paths
base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / "artifacts/embeddings/openalex_T10102_global/embeddings_633ff026.parquet"
backup_path = base_dir / "artifacts/embeddings/openalex_T10102_global/embeddings_633ff026.parquet.bak"
report_path = base_dir / "artifacts/embeddings/openalex_T10102_global/filter_report_633ff026.json"

print(f"Target file: {file_path}")

if not file_path.exists():
    print(f"Error: {file_path} does not exist!")
    sys.exit(1)

# Import filter_by_aspects from edel
sys.path.insert(0, str(base_dir))
from edel.pipeline.embedding import filter_by_aspects

# Read dataframe
print("Reading parquet file...")
df = pd.read_parquet(file_path)
print(f"Initial shape: {df.shape}")

# Backup the original if not already backed up
if not backup_path.exists():
    print(f"Creating backup at: {backup_path}")
    shutil.copy2(file_path, backup_path)
else:
    print(f"Backup already exists at: {backup_path}")

# Filter results and get the report
print("Filtering results...")
df_filtered, report = filter_by_aspects(df)

# Print summary
print("Filtering finished.")
print(f"Final shape: {df_filtered.shape}")
print(f"Filtered rows: {report['total_filtered']} / {report['initial_count']}")
print("Aspect Coverage Details:")
for aspect, stats in report["aspect_coverage"].items():
    print(f"  - {aspect}: filtered={stats['filtered']}, stayed={stats['stayed']}")

# Save filtered version
print(f"Saving new version to: {file_path}")
df_filtered.to_parquet(file_path, index=False)

# Save report
print(f"Saving report to: {report_path}")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("Done successfully!")
