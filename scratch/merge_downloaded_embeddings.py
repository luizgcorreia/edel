import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def main():
    base_path = Path("artifacts")
    emb_dir = base_path / "embeddings" / "openalex_T10102_global"
    emb_path = emb_dir / "embeddings_633ff026.parquet"
    emb_backup_path = emb_dir / "embeddings_633ff026_shuffled.parquet"
    sa_path = base_path / "structured_abstracts" / "openalex_T10102_global" / "sa_b31b87bd.parquet"
    batch_output_path = emb_dir / "batch_output.jsonl"

    print("Step 1: Loading structured abstracts and old embeddings...")
    df_sa = pd.read_parquet(sa_path)
    df_sa_en = df_sa[df_sa["language"] == "en"].copy()
    print(f"Loaded {len(df_sa_en)} English structured abstracts.")

    df_emb_shuffled = pd.read_parquet(emb_backup_path)
    print(f"Loaded {len(df_emb_shuffled)} shuffled embeddings.")

    print("\nStep 2: Building lookup mapping from shuffled embeddings...")
    emb_lookup = {}
    for idx, row in tqdm(df_emb_shuffled.iterrows(), total=len(df_emb_shuffled), desc="Caching shuffled embeddings"):
        has_emb = any(pd.notna(row[f"{field}_embedding"]) for field in ["problem", "method", "finding", "interpretation"])
        if has_emb:
            key = (row["problem"], row["method"], row["finding"], row["interpretation"])
            emb_lookup[key] = {
                f"{field}_embedding": row[f"{field}_embedding"]
                for field in ["problem", "method", "finding", "interpretation"]
            }

    print(f"Cached {len(emb_lookup)} embedding tuples.")

    # Initialize embedding columns in df_sa_en
    for field in ["problem", "method", "finding", "interpretation"]:
        df_sa_en[f"{field}_embedding"] = None

    # Classify papers
    empty_indices = []
    found_indices = []
    missing_indices = []

    for idx, row in df_sa_en.iterrows():
        p, m, f, i = row["problem"], row["method"], row["finding"], row["interpretation"]
        if not p and not m and not f and not i:
            empty_indices.append(idx)
            continue
        
        key = (p, m, f, i)
        if key in emb_lookup:
            found_indices.append(idx)
            for field in ["problem", "method", "finding", "interpretation"]:
                df_sa_en.at[idx, f"{field}_embedding"] = emb_lookup[key][f"{field}_embedding"]
        else:
            missing_indices.append(idx)

    print(f"\nClassification results:")
    print(f"- Empty segments (no embeddings needed): {len(empty_indices)}")
    print(f"- Found and mapped from shuffled file: {len(found_indices)}")
    print(f"- Missing (require API call): {len(missing_indices)}")

    # Get the original indices of the missing ones, matching the df_missing reset_index(drop=True)
    df_missing = df_sa_en.loc[missing_indices].copy()
    original_indices = df_missing.index.tolist()

    print(f"\nStep 3: Parsing batch output file {batch_output_path}...")
    # Read the batch output line by line
    count = 0
    with open(batch_output_path, "r") as f:
        for line in tqdm(f, desc="Reading batch_output.jsonl"):
            data = json.loads(line)
            custom_id = data.get("custom_id")
            if not custom_id:
                continue
            
            # custom_id format: emb::{idx}::{field}
            parts = custom_id.split("::")
            if len(parts) != 3 or parts[0] != "emb":
                continue
            
            idx = int(parts[1])
            field = parts[2]
            
            # Extract embedding
            if data.get("response") and data["response"].get("status_code") == 200:
                body = data["response"]["body"]
                embedding = body["data"][0]["embedding"]
                emb_str = json.dumps(embedding)
                
                orig_idx = original_indices[idx]
                df_sa_en.at[orig_idx, f"{field}_embedding"] = emb_str
                count += 1
            else:
                print(f"Warning: failed request for {custom_id}: {data.get('error')}")

    print(f"Loaded {count} embeddings from batch output.")

    print("\nStep 4: Verifying repaired embeddings...")
    unmapped_count = 0
    for idx, row in df_sa_en.iterrows():
        p, m, f, i = row["problem"], row["method"], row["finding"], row["interpretation"]
        if not p and not m and not f and not i:
            continue
        
        for field in ["problem", "method", "finding", "interpretation"]:
            val = row[field]
            emb = row[f"{field}_embedding"]
            if val and (emb is None or pd.isna(emb)):
                unmapped_count += 1
                break

    print(f"Verification: {unmapped_count} non-empty papers are still unmapped/unembedded.")
    if unmapped_count > 0:
        raise ValueError("Error: Some papers with non-empty segments are still missing embeddings!")

    print(f"\nStep 5: Saving repaired embeddings artifact to {emb_path}...")
    df_sa_en.to_parquet(emb_path, index=False)
    print("Repaired embeddings saved successfully!")

if __name__ == "__main__":
    main()
