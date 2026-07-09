from pathlib import Path
import pandas as pd
import pickle

from edel.dashboard.cache import get_results_df
from edel.dashboard.callbacks.hypothesis_callbacks import _load_features, _get_or_compute_h3_moran_features

base_path = Path("artifacts")
df = get_results_df(base_path)
print("DF columns:", list(df.columns))
print("DF experiment IDs:", list(df["experiment_id"]))

hyp_id = "scientometrics_full_umap"
ctrl_id = "afp_baseline"

hyp_rows = df[df["experiment_id"] == hyp_id]
ctrl_rows = df[df["experiment_id"] == ctrl_id]

print(f"Hyp rows count: {len(hyp_rows)}, Ctrl rows count: {len(ctrl_rows)}")

hyp_metrics = hyp_rows.iloc[0].to_dict()
ctrl_metrics = ctrl_rows.iloc[0].to_dict()

feat_hyp = _load_features(hyp_id, base_path)
feat_ctrl = _load_features(ctrl_id, base_path)

print("feat_hyp loaded?", feat_hyp is not None)
print("feat_ctrl loaded?", feat_ctrl is not None)

moran_hyp = _get_or_compute_h3_moran_features(hyp_id, feat_hyp, base_path)
moran_ctrl = _get_or_compute_h3_moran_features(ctrl_id, feat_ctrl, base_path)
print("moran_hyp computed?", moran_hyp is not None)
print("moran_ctrl computed?", moran_ctrl is not None)
