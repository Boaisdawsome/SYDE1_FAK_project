import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../outputs/merged_biomarkers.csv")

# Focus on SYDE1 features only
syde_cols = [c for c in df.columns if "SYDE1" in c or "ENSG" in c]
syde_df = df[["ModelID"] + syde_cols].dropna(how="all")

print("SYDE1 columns:", syde_cols)
print("Shape:", syde_df.shape)

# Plot expression distribution
plt.hist(df["SYDE1 (85360)"].dropna(), bins=50)
plt.title("SYDE1 Expression (TPM log1p)")
plt.xlabel("Expression level")
plt.ylabel("Number of models")
plt.savefig("../outputs/syde1_expression_hist.png", dpi=300)
plt.close()
print("Saved histogram to outputs/syde1_expression_hist.png")

# Mutation and CNV prevalence
mut_count = df["SYDE1 (85360)_mut_dmg"].sum()
cnv_count = df["SYDE1 (85360)_cnv"].sum()
print(f"SYDE1 mutations in {mut_count} models")
print(f"SYDE1 CNV alterations in {cnv_count} models")