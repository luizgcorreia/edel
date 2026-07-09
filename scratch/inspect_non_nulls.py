import pandas as pd
import numpy as np

path = "/home/correia/edel/artifacts/embeddings/lexicon_null_none_global/embeddings_f6fcb4e3.parquet"
df = pd.read_parquet(path)

# Let's see which rows are not null for problem_embedding
non_null_mask = df["problem_embedding"].notnull()
print(f"Number of non-null problem_embeddings: {non_null_mask.sum()}")

non_null_df = df[non_null_mask]
print("\nNon-null rows:")
for idx, row in non_null_df.iterrows():
    print(f"Row {idx}: ID={row.get('id')}, Title={row.get('title')}")
    # Print abstract text if available
    print(f"Abstract: {row.get('abstract_text')}")
    # Print the structured aspects
    print(f"Problem: {row.get('problem')}")
    print(f"Method: {row.get('method')}")
    print(f"Finding: {row.get('finding')}")
    print(f"Interpretation: {row.get('interpretation')}")
    print("-" * 50)
