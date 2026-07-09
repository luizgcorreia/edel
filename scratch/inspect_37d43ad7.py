import pandas as pd

path = "/home/correia/edel/artifacts/embeddings/lexicon_null_none_global/embeddings_37d43ad7.parquet"
df = pd.read_parquet(path)
print("Columns in sa:")
print(df[['problem', 'method', 'finding', 'interpretation']].head(5))

# Let's count null/empty strings
for col in ['problem', 'method', 'finding', 'interpretation']:
    print(f"{col}: null={df[col].isnull().sum()}, empty={(df[col] == '').sum()}, len_first={len(df[col].iloc[0])}")
