import pandas as pd
import numpy as np

# Simulate the data
df = pd.DataFrame({
    'proj_problem_umap_x': np.random.randn(36374),
    'proj_problem_umap_y': np.random.randn(36374)
})
out_df = df.copy()

# The mask
mask = (out_df['proj_problem_umap_x'] > 2.0)
print(f"Mask leaves {mask.sum()} items")

out_df = out_df[mask.values].copy().reset_index(drop=True)

# Simulate labels
labels = np.zeros(len(out_df))

print(f"out_df length: {len(out_df)}, labels length: {len(labels)}")
out_df["cluster_domain"] = labels
print("Assigned to out_df successfully.")

# What if we assign to the original df?
try:
    df["cluster_domain"] = labels
except Exception as e:
    print(f"Assigning to df failed: {e}")

# What if out_df is 92, and labels is 36374?
labels_big = np.zeros(len(df))
try:
    out_df["cluster_domain"] = labels_big
except Exception as e:
    print(f"Assigning big labels to out_df failed: {e}")

