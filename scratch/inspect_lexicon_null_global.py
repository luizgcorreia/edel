import glob
import pandas as pd
import numpy as np

# Print files under structured_abstracts
print("Structured Abstracts:")
sa_files = glob.glob("/home/correia/edel/artifacts/structured_abstracts/lexicon_null_none_global/*.parquet")
for f in sa_files:
    df = pd.read_parquet(f)
    print(f"File: {f}, Shape: {df.shape}")
    # Show columns like problem, method, finding, interpretation
    cols = [c for c in df.columns if any(a in c for a in ['problem', 'method', 'finding', 'interpretation'])]
    print(f"Columns: {cols}")
    if 'problem' in df.columns:
        print(f"Null problem: {df['problem'].isnull().sum()}")
        print(f"Empty problem: {(df['problem'] == '').sum()}")
        print(f"Sample problem: {df['problem'].iloc[0] if len(df) > 0 else 'None'}")

# Print files under embeddings
print("\nEmbeddings:")
emb_files = glob.glob("/home/correia/edel/artifacts/embeddings/lexicon_null_none_global/*.parquet")
for f in emb_files:
    df = pd.read_parquet(f)
    print(f"File: {f}, Shape: {df.shape}")
    cols = [c for c in df.columns if "embedding" in c]
    print(f"Embedding columns: {cols}")
    for col in cols:
        null_count = df[col].isnull().sum()
        # check if it is all zeros, None, or NaNs
        non_null = df[col].dropna()
        is_zeros = False
        sample_type = None
        if len(non_null) > 0:
            sample = non_null.iloc[0]
            sample_type = type(sample)
            if isinstance(sample, (list, np.ndarray)):
                is_zeros = np.all(np.array(sample) == 0.0)
            elif isinstance(sample, str):
                is_zeros = "0" in sample and len(sample) < 10
        print(f"  {col}: null={null_count}, non_null_len={len(non_null)}, sample_type={sample_type}, is_zeros={is_zeros}")

# Print files under dimensionality reduction
print("\nDimensionality Reduction:")
dr_files = glob.glob("/home/correia/edel/artifacts/dimensionality_reduction/lexicon_null_none_global/*.parquet")
for f in dr_files:
    df = pd.read_parquet(f)
    print(f"File: {f}, Shape: {df.shape}")
    # find coords columns
    coord_cols = [c for c in df.columns if "proj" in c or "x" in c or "y" in c]
    print(f"Columns: {df.columns.tolist()[:15]} ... {df.columns.tolist()[-10:]}")
    for col in coord_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            print(f"  {col}: null={null_count}, Mean={df[col].mean()}, Std={df[col].std()}")
