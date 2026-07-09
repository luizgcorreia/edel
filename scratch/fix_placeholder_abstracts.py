import os
import pickle
import json
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
        df, stats = pickle.load(f)

    papers = ['CRYSTALS-Kyber', 'Karatsuba_Sqrt', 'Polynomial_Interpolation', 'Virtual_Substitution']
    df_papers = df[df['id'].isin(papers)]
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

    for idx, row in df_papers.iterrows():
        paper_id = row['id']
        title = row['title']
        abstract = row['abstract_text']
        print(f"\nProcessing {paper_id}...")

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
            print(f"LLM Response for {paper_id}: {json.dumps(data, indent=2)}")

            # Update row in df
            for key in ["problem", "method", "finding", "interpretation"]:
                snippet = data.get(key, "")
                if snippet == "UNKNOWN":
                    snippet = ""

                # Extract original_val by looking for "\nabstract:\n"
                current_val = row[key]
                if pd.isna(current_val):
                    current_val = ""

                if "\nabstract:\n" in current_val:
                    original_val = current_val.split("\nabstract:\n")[0]
                elif current_val.startswith("abstract:\n"):
                    original_val = ""
                else:
                    original_val = current_val

                # Re-merge
                if snippet:
                    new_val = f"{original_val}\nabstract:\n{snippet}".strip()
                else:
                    new_val = original_val.strip()

                df.at[idx, key] = new_val
                print(f"Updated {key} for {paper_id}.")

        except Exception as e:
            print(f"Error processing {paper_id}: {e}")

    # Save Stage 2 artifact
    with open(pkl_path, "wb") as f:
        pickle.dump((df, stats), f)
    print(f"\nStage 2 artifact saved back to {pkl_path}")

if __name__ == "__main__":
    main()
