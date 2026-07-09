import os
import json
import pickle
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    client = OpenAI(api_key=api_key)

    # 1. Load updated Stage 2 DataFrame
    sa_path = "artifacts/structured_abstracts/afp_isabelle_global/sa_5a64343e.pkl"
    with open(sa_path, "rb") as f:
        df_sa, _ = pickle.load(f)

    # 2. Load Stage 3 DataFrame
    emb_path = "artifacts/embeddings/afp_isabelle_global/embeddings_b463f04c.parquet"
    if not os.path.exists(emb_path):
        print(f"Error: Embeddings parquet file not found at {emb_path}")
        return

    df_emb = pd.read_parquet(emb_path)
    print(f"Loaded embeddings DataFrame with shape {df_emb.shape}")

    papers = ['CRYSTALS-Kyber', 'Karatsuba_Sqrt', 'Polynomial_Interpolation', 'Virtual_Substitution']
    aspects = ["problem", "method", "finding", "interpretation"]

    for paper_id in papers:
        print(f"\nEmbedding aspects for {paper_id}...")
        row_sa = df_sa[df_sa["id"] == paper_id].iloc[0]
        
        # Find index in df_emb
        emb_idx_list = df_emb[df_emb["id"] == paper_id].index
        if len(emb_idx_list) == 0:
            print(f"Warning: {paper_id} not found in embeddings DataFrame.")
            continue
        idx_emb = emb_idx_list[0]

        # Update columns
        df_emb.at[idx_emb, "problem"] = row_sa["problem"]
        df_emb.at[idx_emb, "method"] = row_sa["method"]
        df_emb.at[idx_emb, "finding"] = row_sa["finding"]
        df_emb.at[idx_emb, "interpretation"] = row_sa["interpretation"]

        for aspect in aspects:
            text = str(row_sa[aspect]).strip()
            col_name = f"{aspect}_embedding"
            
            if not text:
                print(f"  {aspect} is empty, setting embedding to None.")
                df_emb.at[idx_emb, col_name] = None
                continue

            try:
                # Call embedding API
                response = client.embeddings.create(
                    input=[text.replace("\n", " ")],
                    model="text-embedding-ada-002"
                )
                emb_vector = response.data[0].embedding
                # Store as JSON string, mirroring pipeline behavior
                df_emb.at[idx_emb, col_name] = json.dumps(emb_vector)
                print(f"  Generated embedding for {aspect} (dim: {len(emb_vector)}).")
            except Exception as e:
                print(f"  Error embedding {aspect} for {paper_id}: {e}")

    # Save Stage 3 output
    df_emb.to_parquet(emb_path, index=False)
    print(f"\nEmbeddings saved back to {emb_path}")

if __name__ == "__main__":
    main()
