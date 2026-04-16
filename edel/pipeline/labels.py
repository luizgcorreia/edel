"""Cluster/label mapping helpers."""


def apply_cluster_labels(df, cluster_labels_df, cluster_col: str):
    """Map cluster IDs in ``cluster_col`` to proposed labels when available."""
    if cluster_labels_df is None:
        return df[cluster_col].astype(str)

    if cluster_col not in df:
        return None

    cluster_type = cluster_col.replace("cluster_", "")
    mapping = {}

    for _, row in cluster_labels_df.iterrows():
        if row["cluster_type"] == cluster_type:
            mapping[row["cluster"]] = row["proposed_label"]

    labeled = df[cluster_col].map(mapping)
    labeled = labeled.fillna(df[cluster_col].astype(str))
    return labeled
