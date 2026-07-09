import pandas as pd

def main():
    path = "/home/correia/edel/artifacts/data_collection/afp_rag_isabelle_global/dataset_bfedcd65.parquet"
    df = pd.read_parquet(path)
    
    print("--- DataFrame Info ---")
    print(df.info())
    print("\n--- Columns ---")
    print(df.columns.tolist())
    
    print("\n--- First 3 Rows ---")
    for idx, row in df.head(3).iterrows():
        print(f"\nRow {idx}:")
        for col in df.columns:
            val = str(row[col])
            if len(val) > 200:
                val = val[:200] + "..."
            print(f"  {col}: {val}")
            
    print("\n--- Character Length Statistics ---")
    text_cols = ["problem", "method", "finding", "interpretation", "statement_text", "proof_text"]
    for col in text_cols:
        if col in df.columns:
            lengths = df[col].astype(str).str.len()
            print(f"\nCol: {col}")
            print(f"  Min length: {lengths.min()}")
            print(f"  Max length: {lengths.max()}")
            print(f"  Mean length: {lengths.mean():.2f}")
            print(f"  Median length: {lengths.median()}")

if __name__ == "__main__":
    main()
