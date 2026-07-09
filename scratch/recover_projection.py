import pickle
import pandas as pd
from pathlib import Path

def recover_artifact(pkl_path):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"File not found: {pkl_path}")
        return

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, tuple) or len(obj) != 2:
        print("The file does not seem to contain the (df, report) tuple.")
        return

    df, report = obj
    
    # 1. Save the DataFrame as Parquet
    parquet_path = pkl_path.with_suffix(".parquet")
    print(f"Saving DataFrame to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)

    # 2. Save the Report as a separate PKL
    # The name should be report_XXXX instead of dr_XXXX
    report_name = pkl_path.name.replace("dr_", "report_")
    report_path = pkl_path.parent.parent / "report" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Report to {report_path}...")
    with open(report_path, "wb") as f:
        pickle.dump(report, f)

    # 3. Clean up the corrupted file
    print(f"Cleaning up {pkl_path}...")
    pkl_path.unlink()
    
    print("Recovery complete!")

if __name__ == "__main__":
    target = "artifacts/dimensionality_reduction/openalex_T10102_global/dr_188df7cc.pkl"
    recover_artifact(target)
