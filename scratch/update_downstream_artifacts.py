import pickle
import pandas as pd
from pathlib import Path

def main():
    # 1. Load the updated Stage 1 dataset
    stage1_path = Path("artifacts/data_collection/afp_isabelle_global/dataset_850322d6.parquet")
    if not stage1_path.exists():
        print("Stage 1 dataset not found! Please run verification first.")
        return

    df1 = pd.read_parquet(stage1_path)
    # Map from id -> (imports, cited_by_count)
    imports_map = df1.set_index("id")["imports"].to_dict()
    citations_map = df1.set_index("id")["cited_by_count"].to_dict()
    print(f"Loaded {len(df1)} entries from Stage 1 dataset.")

    # Helper function to update a DataFrame in place
    def update_df(df: pd.DataFrame) -> bool:
        if "id" not in df.columns:
            return False
        
        updated = False
        # Update imports
        if "imports" in df.columns:
            df["imports"] = df["id"].map(imports_map)
            updated = True
            
        # Update cited_by_count
        if "cited_by_count" in df.columns:
            df["cited_by_count"] = df["id"].map(citations_map).fillna(0).astype(int)
            updated = True
            
        return updated

    # 2. Iterate and update DataFrames in downstream stages
    stages = ["structured_abstracts", "embeddings", "dimensionality_reduction", "clustering"]
    for stage in stages:
        stage_dir = Path("artifacts") / stage / "afp_isabelle_global"
        if not stage_dir.exists():
            continue
        
        for p in stage_dir.glob("*"):
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
                if update_df(df):
                    df.to_parquet(p, index=False)
                    print(f"Updated DataFrame artifact: {p}")
            elif p.suffix == ".pkl":
                try:
                    with open(p, "rb") as f:
                        obj = pickle.load(f)
                    
                    if isinstance(obj, pd.DataFrame):
                        if update_df(obj):
                            with open(p, "wb") as f:
                                pickle.dump(obj, f)
                            print(f"Updated DataFrame pickle: {p}")
                    elif isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], pd.DataFrame):
                        df = obj[0]
                        if update_df(df):
                            with open(p, "wb") as f:
                                pickle.dump((df, obj[1]), f)
                            print(f"Updated tuple pickle (DataFrame, report): {p}")
                except Exception as e:
                    print(f"Error processing pickle {p}: {e}")

    # 3. Update landscape results (Stage 8)
    output_dir = Path("artifacts/output/afp_isabelle_global")
    if output_dir.exists():
        # Load the updated clustering DataFrame to compute terrain
        clustering_path = Path("artifacts/clustering/afp_isabelle_global/clustering_78289fd9.parquet")
        if not clustering_path.exists():
            print("Clustering DataFrame not found, cannot update landscape terrain.")
            return
            
        df_cls = pd.read_parquet(clustering_path)
        
        for p in output_dir.glob("landscape_results_*.pkl"):
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                
                terrain = obj.get("terrain", {})
                if not terrain:
                    continue
                
                raw_metric = terrain.get("raw_metric", "cited_by_count")
                log_scale = terrain.get("log_scale", True)
                num_bins = terrain["x"].shape[0]
                method = "diffusion" # default method
                
                # Global pad boundaries
                px_col = f"proj_problem_{method}_x" if f"proj_problem_{method}_x" in df_cls.columns else f"proj_{method}_x"
                py_col = f"proj_problem_{method}_y" if f"proj_problem_{method}_y" in df_cls.columns else f"proj_{method}_y"
                x_min, x_max = df_cls[px_col].min(), df_cls[px_col].max()
                y_min, y_max = df_cls[py_col].min(), df_cls[py_col].max()
                dx_pad = (x_max - x_min) * 0.10
                dy_pad = (y_max - y_min) * 0.10
                x_range = (x_min - dx_pad, x_max + dx_pad)
                y_range = (y_min - dy_pad, y_max + dy_pad)
                
                from edel.pipeline.landscape import compute_terrain, compute_cluster_regions
                
                # Recompute terrain with updated cited_by_count
                new_terrain = compute_terrain(
                    df=df_cls,
                    method=method,
                    metric=raw_metric,
                    num_bins=num_bins,
                    sigma=1.5,
                    log_scale=log_scale,
                    x_range=x_range,
                    y_range=y_range,
                    scale=1.0
                )
                new_terrain["max_scatter_points"] = terrain.get("max_scatter_points", 1000)
                new_terrain["random_seed"] = terrain.get("random_seed", 42)
                
                regions = compute_cluster_regions(
                    df=df_cls,
                    method=method,
                    x_range=x_range,
                    y_range=y_range,
                    num_bins=num_bins,
                    zi_grid=new_terrain["z"],
                    min_height=0.02
                )
                if regions:
                    new_terrain["explored_mask"] = regions["explored_mask"]
                    new_terrain["boundaries"] = regions["boundaries"]
                    new_terrain["centroids"] = regions["centroids"]
                    
                obj["terrain"] = new_terrain
                
                with open(p, "wb") as f:
                    pickle.dump(obj, f)
                print(f"Updated landscape results: {p}")
            except Exception as e:
                print(f"Error processing landscape results {p}: {e}")

if __name__ == "__main__":
    main()
