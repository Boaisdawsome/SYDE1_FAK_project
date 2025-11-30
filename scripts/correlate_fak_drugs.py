import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np

print("[INFO] Loading data...")

# load model data
model_df = pd.read_csv("outputs/training_dataset.csv")[["ModelID", "FAK_dependency_score"]]

# load drug data
drug_df = pd.read_csv("outputs/fak_lfc_matrix.csv").rename(columns={"DepMap_ID": "ModelID"})

# merge datasets
merged = pd.merge(model_df, drug_df, on="ModelID", how="inner")
print("[INFO] merged shape BEFORE cleaning:", merged.shape)

# detect drug columns
drug_cols = [c for c in merged.columns if "::HTS" in c]

# drop any rows where a drug value is NaN or inf
merged_clean = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=drug_cols)
print("[INFO] merged shape AFTER cleaning:", merged_clean.shape)

print("\n[RESULTS] Correlations:")
for col in drug_cols:
    pear, _ = pearsonr(merged_clean["FAK_dependency_score"], merged_clean[col])
    spear, _ = spearmanr(merged_clean["FAK_dependency_score"], merged_clean[col])
    print(f"{col}: Pearson={pear:.3f}, Spearman={spear:.3f}")
import numpy as np
from scipy.stats import pearsonr, spearmanr

results = []

for col in drug_cols:
    clean = merged[["FAK_dependency_score", col]].replace([np.inf, -np.inf], np.nan).dropna()
    pear, _ = pearsonr(clean["FAK_dependency_score"], clean[col])
    spear, _ = spearmanr(clean["FAK_dependency_score"], clean[col])

    results.append({
        "drug": col,
        "pearson": pear,
        "spearman": spear
    })

pd.DataFrame(results).to_csv("outputs/drug_fak_correlations.csv", index=False)
print("[SAVED] outputs/drug_fak_correlations.csv")