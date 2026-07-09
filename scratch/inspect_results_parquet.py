import pandas as pd
import numpy as np

path = "/home/correia/edel/artifacts/experiments/results.parquet"
df = pd.read_parquet(path)

print("Columns in results.parquet:")
print(df.columns.tolist())
print("\nShape of df:", df.shape)

# Let's see unique configs or experiments
print("\nUnique experiments/configs:")
if "config" in df.columns:
    print(df["config"].value_counts())
elif "experiment" in df.columns:
    print(df["experiment"].value_counts())
else:
    # Print first few rows to see columns
    print(df.head(2))
