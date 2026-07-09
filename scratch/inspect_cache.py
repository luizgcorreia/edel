import pandas as pd
from edel.dashboard.cache import get_results_df

df = get_results_df("artifacts")
if df.empty:
    print("Results DataFrame is empty!")
else:
    h2_keys = ["pm", "pf", "pi", "mp", "mf", "mi", "fp", "fm", "fi", "ip", "im", "if"]
    labels = ["D(M|p)", "D(F|p)", "D(I|p)", "D(P|m)", "D(F|m)", "D(I|m)", "D(P|f)", "D(M|f)", "D(I|f)", "D(P|i)", "D(M|i)", "D(F|i)"]
    
    records = []
    for k, label in zip(h2_keys, labels):
        col = f"h2_pvalue_{k}"
        rec = {"Transition": label}
        for eid in df["experiment_id"].unique():
            row = df[df["experiment_id"] == eid]
            if not row.empty:
                rec[eid] = row.iloc[0][col]
            else:
                rec[eid] = None
        records.append(rec)
        
    print("\nComparison of H2 p-values for all experiments:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    res_df = pd.DataFrame(records)
    print(res_df.to_string(index=False))
    
    print("\nNumber of significant transitions (p < 0.05) per experiment:")
    for col in res_df.columns:
        if col == "Transition":
            continue
        print(f"  {col}: {sum(res_df[col] < 0.05)}")
