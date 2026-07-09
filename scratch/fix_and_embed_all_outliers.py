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
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    client = OpenAI(api_key=api_key)

    # 1. Load Stage 2 artifact
    pkl_path = "artifacts/structured_abstracts/afp_isabelle_global/sa_5a64343e.pkl"
    if not os.path.exists(pkl_path):
        print(f"Error: Stage 2 pickle file not found at {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        df_sa, stats = pickle.load(f)

    # List of 19 papers that need fixing
    papers = [
        'CakeML', 'DPT-SAT-Solver', 'Difference_Bound_Matrices', 'Earley_Parser',
        'FocusStreamsCaseStudies', 'Formal_Puiseux_Series', 'GPU_Kernel_PL', 'Hermite',
        'LL1_Parser', 'Matrix', 'Relative_Security', 'ResiduatedTransitionSystem2',
        'Restriction_Spaces-Examples', 'Routing', 'Sorted_Rewriting', 'Treaps',
        'VerifyThis2018', 'pGCL', 'LTL_to_DRA'
    ]

    df_papers = df_sa[df_sa['id'].isin(papers)]
    print(f"Found {len(df_papers)} papers to fix in Stage 2 artifact.")

    custom_defs = {
        'problem': 'The formal property, theorem, system, or mathematical problem that the theory aims to formalize or verify.',
        'method': 'The formal constructions used in the theory, including definitions, encodings, logical frameworks, or proof techniques.',
        'finding': 'The main formally proven results, including key lemmas, theorems, or derived properties established in the theory.',
        'interpretation': 'The theoretical context of the work, including related theories, imported libraries, and the broader formalization domain.'
    }

    prompt_template = """You are an expert research assistant. Assume the topic is Isabelle.

You will receive the title, keywords, and abstract of a scientific paper.

Your task is to extract text from the abstract corresponding to four epistemic aspects.

Definitions:

1. Problem / Research Question
{problem_def}

2. Methods / Evidence
{method_def}

3. Findings / Results
{finding_def}

4. Interpretation / Discussion
{interpretation_def}

Rules:

- Only extract text that appears in the original abstract.
- You may extract full sentences or sentence fragments.
- Do NOT paraphrase or invent new text.
- The same text may appear in multiple categories if relevant.
- Prefer partial evidence over returning UNKNOWN.
- Return "UNKNOWN" only if the abstract contains no evidence for that aspect.

Return your answer as valid JSON:

{{
  "problem": "...",
  "method": "...",
  "finding": "...",
  "interpretation": "..."
}}

Title:
{title}

Abstract:
{abstract_text}

Keywords:
[]

JSON Answer:
"""

    structured_results = {}

    for idx, row in df_papers.iterrows():
        paper_id = row['id']
        title = row['title']
        abstract = row['abstract_text']
        print(f"\nProcessing LLM structure for {paper_id}...")

        prompt = prompt_template.format(
            problem_def=custom_defs['problem'],
            method_def=custom_defs['method'],
            finding_def=custom_defs['finding'],
            interpretation_def=custom_defs['interpretation'],
            title=title,
            abstract_text=abstract
        )

        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                response_format={'type': 'json_object'}
            )
            data = json.loads(response.choices[0].message.content)
            print(f"  LLM: problem='{data.get('problem')[:60]}...'")
            structured_results[paper_id] = data

            # Update row in df_sa
            for key in ["problem", "method", "finding", "interpretation"]:
                snippet = data.get(key, "")
                if snippet == "UNKNOWN":
                    snippet = ""

                current_val = row[key]
                if pd.isna(current_val):
                    current_val = ""

                if "\nabstract:\n" in current_val:
                    original_val = current_val.split("\nabstract:\n")[0]
                elif current_val.startswith("abstract:\n"):
                    original_val = ""
                else:
                    original_val = current_val

                if snippet:
                    new_val = f"{original_val}\nabstract:\n{snippet}".strip()
                else:
                    new_val = original_val.strip()

                df_sa.at[idx, key] = new_val

        except Exception as e:
            print(f"Error processing {paper_id}: {e}")

    # Save Stage 2 artifact
    with open(pkl_path, "wb") as f:
        pickle.dump((df_sa, stats), f)
    print(f"\nStage 2 artifact saved back to {pkl_path}")

    # 2. Load Stage 3 Embeddings DataFrame
    emb_path = "artifacts/embeddings/afp_isabelle_global/embeddings_b463f04c.parquet"
    if not os.path.exists(emb_path):
        print(f"Error: Embeddings parquet file not found at {emb_path}")
        return

    df_emb = pd.read_parquet(emb_path)
    print(f"Loaded embeddings DataFrame with shape {df_emb.shape}")

    aspects = ["problem", "method", "finding", "interpretation"]

    for paper_id, data in structured_results.items():
        print(f"\nEmbedding aspects for {paper_id}...")
        
        # Find index in df_emb
        emb_idx_list = df_emb[df_emb["id"] == paper_id].index
        if len(emb_idx_list) == 0:
            print(f"Warning: {paper_id} not found in embeddings DataFrame.")
            continue
        idx_emb = emb_idx_list[0]

        # Get updated texts from Stage 2 df
        row_sa = df_sa[df_sa["id"] == paper_id].iloc[0]

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
                df_emb.at[idx_emb, col_name] = json.dumps(emb_vector)
                print(f"  Generated embedding for {aspect} (dim: {len(emb_vector)}).")
            except Exception as e:
                print(f"  Error embedding {aspect} for {paper_id}: {e}")

    # Save Stage 3 output
    df_emb.to_parquet(emb_path, index=False)
    print(f"\nEmbeddings saved back to {emb_path}")

if __name__ == "__main__":
    main()
