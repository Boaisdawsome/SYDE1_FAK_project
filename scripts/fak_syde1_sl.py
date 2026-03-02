import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

crispr_path = os.path.join(BASE_DIR, "data", "CRISPRGeneEffect.csv")
rna_path = os.path.join(BASE_DIR, "data", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

print("Loading:")
print(crispr_path)
print(rna_path)

# --------------------------------------------------
# 2. Load + Transpose (Genes as rows)
# --------------------------------------------------

crispr_df = pd.read_csv(crispr_path, index_col=0).T
rna_df = pd.read_csv(rna_path, index_col=0).T

# Clean column names
crispr_df.columns = crispr_df.columns.astype(str).str.strip()
rna_df.columns = rna_df.columns.astype(str).str.strip()

# If RNA has suffixes like ACH-000015_LUNG → remove suffix
rna_df.columns = rna_df.columns.str.split("_").str[0]

# --------------------------------------------------
# 3. Align Cell Lines
# --------------------------------------------------

common_cells = list(set(crispr_df.columns).intersection(set(rna_df.columns)))

print("Overlapping cell lines:", len(common_cells))

if len(common_cells) < 5:
    raise ValueError("Not enough overlapping cell lines. Check ID formatting.")

crispr_df = crispr_df[common_cells]
rna_df = rna_df[common_cells]

# --------------------------------------------------
# 4. Extract PTK2 + SYDE1 Dynamically
# --------------------------------------------------

ptk2_candidates = [g for g in crispr_df.index if "PTK2" in str(g)]
syde1_candidates = [g for g in rna_df.index if "SYDE1" in str(g)]

if len(ptk2_candidates) == 0:
    raise ValueError("PTK2 not found in CRISPR dataset.")

if len(syde1_candidates) == 0:
    raise ValueError("SYDE1 not found in RNA dataset.")

ptk2_gene = ptk2_candidates[0]
syde1_gene = syde1_candidates[0]

print("Using PTK2 row:", ptk2_gene)
print("Using SYDE1 row:", syde1_gene)

fak_dep = crispr_df.loc[ptk2_gene]
syde1_expr = rna_df.loc[syde1_gene]

# --------------------------------------------------
# 5. Merge + Clean
# --------------------------------------------------

df = pd.concat([syde1_expr, fak_dep], axis=1)
df.columns = ["SYDE1_expr", "PTK2_dep"]
df = df.dropna()

print("Data points after cleanup:", len(df))

if len(df) < 5:
    raise ValueError("Too few aligned data points after cleanup.")

# --------------------------------------------------
# 6. Compute Correlation
# --------------------------------------------------

r, p = pearsonr(df["SYDE1_expr"], df["PTK2_dep"])

print("\nCorrelation (r):", round(r,4))
print("P-value:", f"{p:.2e}")

# --------------------------------------------------
# 7. Scatter Plot
# --------------------------------------------------

plt.figure(figsize=(7,6))
sns.regplot(x="SYDE1_expr", y="PTK2_dep", data=df, scatter_kws={"alpha":0.6})
plt.title(f"SYDE1 Expression vs PTK2 Dependency\nr={r:.3f}, p={p:.2e}")
plt.xlabel("SYDE1 Expression (TPM log1p)")
plt.ylabel("PTK2 Gene Effect")
plt.tight_layout()

output_path = os.path.join(BASE_DIR, "outputs", "syde1_fak_scatter.png")
plt.savefig(output_path, dpi=300)
plt.show()

print("\nSaved plot to:", output_path)