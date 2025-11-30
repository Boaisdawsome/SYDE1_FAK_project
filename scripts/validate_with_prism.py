import pandas as pd
from scipy.stats import spearmanr, pearsonr

print("[INFO] Loading model predictions & PRISM data...")

# load predictions (training_dataset.csv already has FAK dependency info)
df = pd.read_csv("outputs/training_dataset.csv")

# load PRISM drug sensitivity file (replace this with the real one later)
prism = pd.read_csv("data/PRISM_FAK_inhibitors.csv")

# align identifiers
df = df.rename(columns={"ModelID": "DepMap_ID"})
merged = pd.merge(df, prism, on="DepMap_ID", how="inner")
print("[INFO] merged shape:", merged.shape)

# check correlations
if "FAK_dependency_score" in merged.columns and "AUC" in merged.columns:
    pear, _ = pearsonr(merged["FAK_dependency_score"], merged["AUC"])
    spear, _ = spearmanr(merged["FAK_dependency_score"], merged["AUC"])
    print(f"[RESULTS] Pearson correlation: {pear:.3f}")
    print(f"[RESULTS] Spearman correlation: {spear:.3f}")
else:
    print("Missing required columns (FAK_dependency_score or AUC). Check PRISM file headers.")

# export for later visualization
merged[["FAK_dependency_score", "AUC"]].to_csv("outputs/fak_prism_validation.csv", index=False)
print("Saved outputs/fak_prism_validation.csv")