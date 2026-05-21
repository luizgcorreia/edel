"""Utilities for applying labels to clustered data."""

import pandas as pd


def build_cluster_label_map(cluster_labels_df: pd.DataFrame) -> dict:
    """
    Build a mapping from (cluster_type, cluster_id) to label.
    
    The input DataFrame should have columns: cluster_type, cluster, proposed_label.
    """
    if cluster_labels_df is None:
        return {}

    mapping = {}
    for _, row in cluster_labels_df.iterrows():
        cluster_type = row["cluster_type"]
        cid = row["cluster"]
        label = row.get("proposed_label", str(cid))
        key = (cluster_type, cid)
        mapping[key] = label

    return mapping


def apply_cluster_labels(
    df: pd.DataFrame,
    cluster_labels_df: pd.DataFrame,
    cluster_col: str,
) -> pd.Series:
    """
    Replace cluster IDs in df[cluster_col] with the labels stored in cluster_labels_df.
    
    cluster_labels_df must contain:
        cluster_type
        cluster
        proposed_label
    """
    # If no labels, return original column as string
    if cluster_labels_df is None:
        return df[cluster_col].astype(str)

    if cluster_col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    # e.g., cluster_domain -> domain
    cluster_type = cluster_col.replace("cluster_", "")

    # build mapping for this specific cluster type
    mapping = {}
    for _, row in cluster_labels_df.iterrows():
        if row["cluster_type"] == cluster_type:
            mapping[row["cluster"]] = row["proposed_label"]

    # apply mapping
    labeled = df[cluster_col].map(mapping)

    # keep original id if label missing
    labeled = labeled.fillna(df[cluster_col].astype(str))

    return labeled
