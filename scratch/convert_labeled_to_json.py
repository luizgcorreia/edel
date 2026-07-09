import pickle
import json
from pathlib import Path

pkl_path = Path("artifacts/labeling/openalex_T10102_global/labeled_a4a2c014.pkl")
json_path = pkl_path.with_suffix(".json")

print(f"Reading {pkl_path}...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"Writing {json_path}...")
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("Done!")
