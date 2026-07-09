import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Adjust path to find edel package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edel.io.llm import VoyageClient

load_dotenv()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def run_experiments():
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("[ERROR] VOYAGE_API_KEY is not set in environment. Exiting.")
        return
        
    client = VoyageClient(model="voyage-code-3", api_key=api_key)
    
    # Load dataset
    df = pd.read_parquet('/home/correia/edel/artifacts/rag_index/metadata.parquet')
    
    # Select 8 lemmas
    target_titles = [
        'HOL-Library.Multiset.multiset_eq_iff',
        'HOL-Library.Multiset.mset_subset_eq_mono_add',
        'HOL-Library.Multiset.size_empty',
        'HOL-Library.Multiset.union_commute',
        'HOL-Library.Multiset.union_assoc',
        'HOL-Library.Multiset.add_mset_commute',
        'HOL-Library.Multiset.count_single',
        'HOL-Library.Multiset.diff_empty'
    ]
    
    selected_df = df[df['title'].isin(target_titles)].copy()
    print(f"Selected {len(selected_df)} lemmas for analysis.")
    
    # Generate representations for Format A, B, C
    # Format A: aspect-only combined (expanded aspects)
    def make_format_a(row):
        return (
            f"Problem: {row['problem']}\n"
            f"Method: {row['method']}\n"
            f"Finding: {row['finding']}\n"
            f"Interpretation: {row['interpretation']}"
        )
        
    # Format B: raw statement and proof
    def make_format_b(row):
        proof = row['proof_text'] if row['proof_text'] else "No proof body available (definition or abbreviation)."
        return (
            f"Statement:\n{row['statement_text']}\n"
            f"Proof:\n{proof}"
        )
        
    # Format C: structured semantic synthesis
    def make_format_c(row):
        proof = row['proof_text'] if row['proof_text'] else "No proof body available (definition or abbreviation)."
        return (
            f"Title: {row['title']}\n"
            f"Theory Context: {row['method']}\n"
            f"Extracted Aspects:\n"
            f"- Statement: {row['problem']}\n"
            f"- Strategy: {row['finding']}\n"
            f"- Dependencies: {row['interpretation']}\n"
            f"Source Code:\n"
            f"- Declaration: {row['statement_text']}\n"
            f"- Proof Body: {proof}"
        )
        
    selected_df['format_A'] = selected_df.apply(make_format_a, axis=1)
    selected_df['format_B'] = selected_df.apply(make_format_b, axis=1)
    selected_df['format_C'] = selected_df.apply(make_format_c, axis=1)
    
    # Define queries and their target titles
    queries = [
        {
            "id": "Q1",
            "type": "Natural Language",
            "query": "prove that two multisets are equal if they have the same element count",
            "target": "HOL-Library.Multiset.multiset_eq_iff"
        },
        {
            "id": "Q2",
            "type": "Code Subgoal",
            "query": "(⋀x. count A x = count B x) ⟹ A = B",
            "target": "HOL-Library.Multiset.multiset_eq_iff"
        },
        {
            "id": "Q3",
            "type": "Natural Language",
            "query": "monotonicity of multiset union with respect to ordering",
            "target": "HOL-Library.Multiset.mset_subset_eq_mono_add"
        },
        {
            "id": "Q4",
            "type": "Code Subgoal",
            "query": "A <# B ⟹ C + A <# C + B",
            "target": "HOL-Library.Multiset.mset_subset_eq_mono_add"
        },
        {
            "id": "Q5",
            "type": "Natural Language",
            "query": "the size of a multiset is zero if and only if it is empty",
            "target": "HOL-Library.Multiset.size_empty"
        },
        {
            "id": "Q6",
            "type": "Code Subgoal",
            "query": "size M = 0 ⟷ M = {#}",
            "target": "HOL-Library.Multiset.size_empty"
        }
    ]
    
    # Print length statistics
    print("\n--- Representation Lengths (Characters) ---")
    for fmt in ['format_A', 'format_B', 'format_C']:
        lens = selected_df[fmt].apply(len)
        print(f"{fmt}: Mean={lens.mean():.1f}, Median={lens.median():.1f}, Min={lens.min()}, Max={lens.max()}")
        
    # Embed everything
    print("\nGenerating Voyage embeddings...")
    
    # Embed queries (use input_type='query')
    query_texts = [q["query"] for q in queries]
    query_embs = client.generate_embedding(query_texts, input_type="query")
    for q, emb in zip(queries, query_embs):
        q["embedding"] = emb
        
    # Embed lemmas for each format (use input_type='document')
    lemma_embs = {}
    for fmt in ['format_A', 'format_B', 'format_C']:
        texts = selected_df[fmt].tolist()
        embs = client.generate_embedding(texts, input_type="document")
        lemma_embs[fmt] = {title: emb for title, emb in zip(selected_df['title'].tolist(), embs)}
        
    print("Embeddings generated. Computing similarities...")
    
    results = []
    
    for q in queries:
        q_emb = q["embedding"]
        target_title = q["target"]
        
        q_res = {
            "query_id": q["id"],
            "query_type": q["type"],
            "query_text": q["query"],
            "target_title": target_title,
            "formats": {}
        }
        
        for fmt in ['format_A', 'format_B', 'format_C']:
            # Calculate similarities for all 8 lemmas
            sims = {}
            for title in target_titles:
                sims[title] = float(cosine_similarity(q_emb, lemma_embs[fmt][title]))
                
            # Rank lemmas
            sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            ranks = {title: rank for rank, (title, _) in enumerate(sorted_sims, 1)}
            
            target_sim = sims[target_title]
            target_rank = ranks[target_title]
            top_1_title, top_1_sim = sorted_sims[0]
            
            # Non-target similarities
            non_target_sims = [sim for title, sim in sims.items() if title != target_title]
            avg_non_target = float(np.mean(non_target_sims))
            
            # SNR
            snr = target_sim / avg_non_target if avg_non_target > 0 else 0
            
            q_res["formats"][fmt] = {
                "target_similarity": target_sim,
                "target_rank": target_rank,
                "top_1_title": top_1_title,
                "top_1_similarity": top_1_sim,
                "avg_non_target_similarity": avg_non_target,
                "signal_to_noise_ratio": snr,
                "all_similarities": sims
            }
            
        results.append(q_res)
        
    # Print results summary table
    print("\n" + "="*80)
    print(f"{'Query ID':<10} | {'Type':<15} | {'Format':<10} | {'Target Sim':<10} | {'Target Rank':<12} | {'Top 1 Sim':<10} | {'SNR':<6}")
    print("="*80)
    for r in results:
        for fmt in ['format_A', 'format_B', 'format_C']:
            f_data = r["formats"][fmt]
            print(f"{r['query_id']:<10} | {r['query_type']:<15} | {fmt:<10} | {f_data['target_similarity']:.4f}     | {f_data['target_rank']:<12} | {f_data['top_1_similarity']:.4f}    | {f_data['signal_to_noise_ratio']:.2f}")
        print("-"*80)
        
    # Print aggregates
    print("\n=== Aggregate Metrics across all queries ===")
    for fmt in ['format_A', 'format_B', 'format_C']:
        avg_target_sim = np.mean([r["formats"][fmt]["target_similarity"] for r in results])
        avg_rank = np.mean([r["formats"][fmt]["target_rank"] for r in results])
        avg_non_target_sim = np.mean([r["formats"][fmt]["avg_non_target_similarity"] for r in results])
        avg_snr = np.mean([r["formats"][fmt]["signal_to_noise_ratio"] for r in results])
        
        # Rank-1 accuracy
        rank_1_acc = np.mean([1 if r["formats"][fmt]["target_rank"] == 1 else 0 for r in results]) * 100
        
        print(f"\nFormat {fmt[-1]}:")
        print(f"  Mean Target Similarity:      {avg_target_sim:.4f}")
        print(f"  Mean Non-Target Similarity:  {avg_non_target_sim:.4f} (Lower = less noise/higher contrast)")
        print(f"  Mean Target Rank:            {avg_rank:.2f} (Lower is better)")
        print(f"  Rank-1 Retrieval Accuracy:   {rank_1_acc:.1f}%")
        print(f"  Signal-to-Noise Ratio (SNR): {avg_snr:.3f} (Higher is better)")
        
    # Save detailed JSON report
    out_path = Path("/home/correia/edel/scratch/embedding_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed experimental results saved to: {out_path}")

if __name__ == "__main__":
    run_experiments()
