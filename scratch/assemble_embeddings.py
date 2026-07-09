import json
from pathlib import Path
import pandas as pd
from edel.experiments.registry import init_registry, get_experiment
from edel.io.artifact import make_stage_artifact, save_artifact
from edel.io.llm import get_llm_client
from edel.pipeline.embedding import filter_by_aspects

def main():
    base_path = Path("artifacts")
    configs_dir = base_path / "configs"
    
    print("Initializing registry...")
    init_registry(configs_dir)
    config = get_experiment("scigen_null_5K")
    
    # 1. Load the dataset
    art_data = make_stage_artifact(config, base_path, "data_collection", "dataset")
    print(f"Loading dataset from {art_data.parquet_path}...")
    df = pd.read_parquet(art_data.parquet_path)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Apply aspect filtering
    df_filtered, filter_report = filter_by_aspects(df)
    print(f"Filtered dataset: {len(df_filtered)} rows.")
    print("Filter report:", filter_report)
    
    # 2. Combine batch IDs from both logs to be 100% complete
    batch_ids = [
        "batch_6a22e7f1e9cc819086c64e47cf31c15a",
        "batch_6a22ed623e1c8190bd0d56a04e3c6140",
        "batch_6a22f57c0ebc81908a814b7ef14d2bd4",
        "batch_6a22fd953afc8190a8f888cc7f60b4d1",
        "batch_6a23066700fc8190b54bef59364d7d7f"
    ]
    print(f"Batch IDs: {batch_ids}")
    
    # 3. Retrieve batch results from OpenAI
    print("Initializing LLM client...")
    client = get_llm_client(config["embedding"])
    
    all_results = {}
    for b_id in batch_ids:
        print(f"Polling batch {b_id}...")
        status_info = client.poll_batch(b_id)
        status = status_info["status"]
        print(f"Batch {b_id} status: {status}")
        if status == "completed":
            results = status_info["results"]
            if results:
                all_results.update(results)
                print(f"Retrieved {len(results)} results from batch {b_id}. Total so far: {len(all_results)}")
            else:
                print(f"Warning: Batch {b_id} is completed but returned no results.")
        else:
            raise RuntimeError(f"Batch {b_id} is not completed (status: {status}). Please check OpenAI batch status.")
            
    print(f"Total retrieved results: {len(all_results)}")
    
    # 4. Map results back to DataFrame
    fields_to_embed = ["problem", "method", "finding", "interpretation"]
    out = df_filtered.copy()
    
    for field in fields_to_embed:
        target_col = f"{field}_embedding"
        col_data = []
        missing_count = 0
        for idx in out.index:
            key = f"emb::{idx}::{field}"
            res = all_results.get(key)
            if res is not None:
                col_data.append(json.dumps(res))
            else:
                col_data.append(None)
                missing_count += 1
        out[target_col] = col_data
        print(f"Mapped {field}_embedding. Missing count: {missing_count}")
        if missing_count > 0:
            print(f"Warning: {missing_count} rows are missing embeddings for field {field}!")
            
    # 5. Save the final embeddings artifact
    art_emb = make_stage_artifact(config, base_path, "embeddings", "embeddings")
    print(f"Saving final embeddings to {art_emb.parquet_path}...")
    save_artifact(art_emb, out)
    
    # Save the filter report
    art_report = make_stage_artifact(config, base_path, "embeddings", "filter_report")
    print(f"Saving filter report to {art_report.pkl_path}...")
    save_artifact(art_report, filter_report)
    
    print("✅ Embeddings artifact assembly complete!")

if __name__ == "__main__":
    main()
