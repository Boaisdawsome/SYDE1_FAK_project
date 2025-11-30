import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/fak_lfc_matrix.csv")
model_df = pd.read_csv("outputs/training_dataset.csv")[["ModelID", "FAK_dependency_score"]]

# fak_lfc_matrix.csv uses DepMap_ID instead of ModelID
df = df.rename(columns={"DepMap_ID": "ModelID"})

merged = pd.merge(model_df, df, on="ModelID", how="inner")

drug_cols = [c for c in merged.columns if "::HTS" in c]

for col in drug_cols:
    plt.figure(figsize=(6, 4))
    plt.scatter(merged["FAK_dependency_score"], merged[col], s=10, alpha=0.6)
    plt.xlabel("FAK Dependency Score (Chronos)")
    plt.ylabel(f"{col} LFC")
    plt.title(f"Correlation: {col}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    fname = f"outputs/scatter_{col.replace(':','_')}.png"
    plt.savefig(fname, dpi=200)
    print("Saved", fname)

print("All scatterplots generated.")