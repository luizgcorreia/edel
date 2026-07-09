import json
import hashlib
import sqlite3
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

def get_hash(p, m, f, i):
    p = p.strip() if p else ""
    m = m.strip() if m else ""
    f = f.strip() if f else ""
    i = i.strip() if i else ""
    key_str = f"{p}||{m}||{f}||{i}"
    return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

def main():
    base_path = Path("artifacts")
    emb_dir = base_path / "embeddings" / "openalex_T10102_global"
    emb_path = emb_dir / "embeddings_633ff026.parquet"
    emb_shuffled_path = emb_dir / "embeddings_633ff026_shuffled.parquet"
    sa_path = base_path / "structured_abstracts" / "openalex_T10102_global" / "sa_b31b87bd.parquet"
    
    db_path = emb_dir / "merge_temp.db"
    if db_path.exists():
        db_path.unlink()
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    print("Step 1: Creating SQLite tables and indexes...")
    c.execute("""
        CREATE TABLE shuffled_cache (
            hash TEXT PRIMARY KEY,
            problem_embedding TEXT,
            method_embedding TEXT,
            finding_embedding TEXT,
            interpretation_embedding TEXT
        )
    """)
    c.execute("""
        CREATE TABLE new_embeddings (
            orig_idx INTEGER PRIMARY KEY,
            problem_embedding TEXT,
            method_embedding TEXT,
            finding_embedding TEXT,
            interpretation_embedding TEXT
        )
    """)
    conn.commit()

    print("\nStep 2: Loading shuffled embeddings in chunks and populating SQLite cache...")
    pf = pq.ParquetFile(emb_shuffled_path)
    
    # Read group by group to be memory efficient
    for rg_idx in tqdm(range(pf.num_row_groups), desc="Processing shuffled row groups"):
        df_chunk = pf.read_row_group(
            rg_idx, 
            columns=["problem", "method", "finding", "interpretation", 
                     "problem_embedding", "method_embedding", "finding_embedding", "interpretation_embedding"]
        ).to_pandas()
        
        insert_data = []
        for _, row in df_chunk.iterrows():
            p, m, f, i = row["problem"], row["method"], row["finding"], row["interpretation"]
            if not p and not m and not f and not i:
                continue
            
            # Check if any embedding field is non-null
            has_emb = any(pd.notna(row[f"{field}_embedding"]) for field in ["problem", "method", "finding", "interpretation"])
            if has_emb:
                h = get_hash(p, m, f, i)
                insert_data.append((
                    h,
                    row["problem_embedding"] if pd.notna(row["problem_embedding"]) else None,
                    row["method_embedding"] if pd.notna(row["method_embedding"]) else None,
                    row["finding_embedding"] if pd.notna(row["finding_embedding"]) else None,
                    row["interpretation_embedding"] if pd.notna(row["interpretation_embedding"]) else None,
                ))
        
        if insert_data:
            c.executemany("""
                INSERT OR REPLACE INTO shuffled_cache 
                (hash, problem_embedding, method_embedding, finding_embedding, interpretation_embedding)
                VALUES (?, ?, ?, ?, ?)
            """, insert_data)
            conn.commit()
            
    print("Shuffled cache SQLite population done.")

    print("\nStep 3: Loading structured abstracts and identifying missing indices...")
    df_sa = pd.read_parquet(sa_path)
    df_sa_en = df_sa[df_sa["language"] == "en"].copy()
    print(f"Loaded {len(df_sa_en)} English structured abstracts.")

    # Classify english abstracts using SQLite cache to find which are missing
    empty_count = 0
    found_count = 0
    missing_indices = []

    for idx, row in tqdm(df_sa_en.iterrows(), total=len(df_sa_en), desc="Classifying english abstracts"):
        p, m, f, i = row["problem"], row["method"], row["finding"], row["interpretation"]
        if not p and not m and not f and not i:
            empty_count += 1
            continue
            
        h = get_hash(p, m, f, i)
        c.execute("SELECT 1 FROM shuffled_cache WHERE hash = ?", (h,))
        res = c.fetchone()
        if res:
            found_count += 1
        else:
            missing_indices.append(idx)

    print(f"Classification results:")
    print(f"- Empty: {empty_count}")
    print(f"- Found in cache: {found_count}")
    print(f"- Missing (require API): {len(missing_indices)}")

    df_missing = df_sa_en.loc[missing_indices].copy()
    original_indices = df_missing.index.tolist()

    print("\nStep 4: Parsing JSONL batch output files and updating SQLite...")
    jsonl_files = [
        emb_dir / "batch_output.jsonl",
        emb_dir / "batch_output_2.jsonl"
    ]
    
    for jsonl_path in jsonl_files:
        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} does not exist!")
            continue
            
        print(f"Reading {jsonl_path}...")
        with open(jsonl_path, "r") as file_obj:
            for line in tqdm(file_obj, desc=f"Parsing {jsonl_path.name}"):
                if not line.strip():
                    continue
                data = json.loads(line)
                custom_id = data.get("custom_id")
                if not custom_id:
                    continue
                
                # Format: emb::{idx}::{field}
                parts = custom_id.split("::")
                if len(parts) != 3 or parts[0] != "emb":
                    continue
                
                idx = int(parts[1])
                field = parts[2]
                
                if data.get("response") and data["response"].get("status_code") == 200:
                    body = data["response"]["body"]
                    embedding = body["data"][0]["embedding"]
                    emb_str = json.dumps(embedding)
                    
                    orig_idx = original_indices[idx]
                    
                    # Upsert into new_embeddings
                    c.execute("SELECT 1 FROM new_embeddings WHERE orig_idx = ?", (orig_idx,))
                    exists = c.fetchone()
                    if exists:
                        c.execute(f"UPDATE new_embeddings SET {field}_embedding = ? WHERE orig_idx = ?", (emb_str, orig_idx))
                    else:
                        c.execute(f"""
                            INSERT INTO new_embeddings (orig_idx, {field}_embedding)
                            VALUES (?, ?)
                        """, (orig_idx, emb_str))
                else:
                    print(f"Warning: Failed request for {custom_id}: {data.get('error')}")
            conn.commit()

    print("\nStep 5: Constructing final embeddings parquet...")
    # Add empty columns to df_sa_en
    for field in ["problem_embedding", "method_embedding", "finding_embedding", "interpretation_embedding"]:
        df_sa_en[field] = None

    # Retrieve all embeddings and merge them
    for idx, row in tqdm(df_sa_en.iterrows(), total=len(df_sa_en), desc="Merging embeddings"):
        p, m, f, i = row["problem"], row["method"], row["finding"], row["interpretation"]
        if not p and not m and not f and not i:
            continue
            
        # Check if in new_embeddings
        c.execute("SELECT problem_embedding, method_embedding, finding_embedding, interpretation_embedding FROM new_embeddings WHERE orig_idx = ?", (idx,))
        res = c.fetchone()
        if res:
            df_sa_en.at[idx, "problem_embedding"] = res[0]
            df_sa_en.at[idx, "method_embedding"] = res[1]
            df_sa_en.at[idx, "finding_embedding"] = res[2]
            df_sa_en.at[idx, "interpretation_embedding"] = res[3]
        else:
            h = get_hash(p, m, f, i)
            c.execute("SELECT problem_embedding, method_embedding, finding_embedding, interpretation_embedding FROM shuffled_cache WHERE hash = ?", (h,))
            res = c.fetchone()
            if res:
                df_sa_en.at[idx, "problem_embedding"] = res[0]
                df_sa_en.at[idx, "method_embedding"] = res[1]
                df_sa_en.at[idx, "finding_embedding"] = res[2]
                df_sa_en.at[idx, "interpretation_embedding"] = res[3]

    print("\nStep 6: Verifying repaired embeddings...")
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

    print(f"\nStep 7: Saving repaired embeddings parquet to {emb_path}...")
    df_sa_en.to_parquet(emb_path, index=False)
    print("Repaired embeddings saved successfully!")
    
    # Close SQLite and clean up temp file
    conn.close()
    if db_path.exists():
        db_path.unlink()

if __name__ == "__main__":
    main()
