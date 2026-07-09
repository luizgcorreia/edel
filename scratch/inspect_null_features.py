import pickle
import pandas as pd
import numpy as np

path = "/home/correia/edel/artifacts/experiments/afp_lexicon_null/features.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print("Type of data:", type(data))
if isinstance(data, dict):
    print("Keys in data:")
    for k, v in data.items():
        print(f"  {k}: type={type(v)}")
        if isinstance(v, pd.DataFrame):
            print(f"    DataFrame columns: {v.columns.tolist()}")
            print(f"    DataFrame shape: {v.shape}")
        elif isinstance(v, (list, np.ndarray)):
            print(f"    Length: {len(v)}")
else:
    print(data)
