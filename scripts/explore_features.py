import os
import pandas as pd
import numpy as np

"""
EDA Step 1 for SYDE1_CANCER_RESEARCH

What this does:
1) Loads outputs/merged_biomarkers.csv
2) Prints and saves core dataset stats:
   - number of models (rows) and features (columns)
   - % missing per column; flags columns with >20% missing
   - constant/near-constant columns
3) Identifies likely binary features (mutation/CNV) vs. continuous (expression)
4) Computes simple summaries:
   - mutation/CNV prevalence (mean of 0/1 columns)
   - expression variance ranking (top-N)
5) Optionally checks for SYDE1 column presence
6) Writes text + CSV summaries to outputs/

Run:
python scripts/explore_features.py
"""

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
INPUT = os.path.join(PROJECT_ROOT, "outputs", "merged_biomarkers.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading merged dataset...")
df = pd.read_csv(INPUT, index_col=0)

# Drop completely empty rows
df = df.dropna(how="all")
if df.empty:
    print("Warning: merged_biomarkers.csv is empty after dropping NaNs.")
else:
    print(f"Dataset loaded successfully with {len(df)} rows.")

# Basic shape
n_models, n_features = df.shape
print(f"Rows (models): {n_models}")
print(f"Columns (features): {n_features}")

# Missingness per column
print("Computing missingness...")
missing_pct = df.isna().mean() * 100.0
high_missing = missing_pct[missing_pct > 20.0].sort_values(ascending=False)

# Constant / near-constant columns
print("Detecting low-variance features...")
nunique = df.nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()

# Identify binary vs continuous
def is_binary_like(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) == 0:
        return False
    numeric_vals = pd.to_numeric(vals, errors="coerce").dropna().astype(float)
    return set(numeric_vals.unique()).issubset({0.0, 1.0})

print("Classifying features into binary-like vs continuous...")
binary_mask = df.apply(is_binary_like, axis=0)
binary_cols = df.columns[binary_mask].tolist()
cont_cols = df.columns[~binary_mask].tolist()

# Summaries
print("Summarizing binary-like features (mutation/CNV prevalence)...")
if binary_cols:
    binary_summary = pd.DataFrame({
        "feature": binary_cols,
        "prevalence": df[binary_cols].mean(numeric_only=True)
    }).sort_values("prevalence", ascending=False)
else:
    binary_summary = pd.DataFrame(columns=["feature", "prevalence"])

print("Summarizing continuous features (variance ranking)...")
if cont_cols:
    cont_df = df[cont_cols].apply(pd.to_numeric, errors="coerce")
    variance = cont_df.var(numeric_only=True)
    top_var = variance.sort_values(ascending=False).head(200)
    top_var_df = top_var.reset_index()
    top_var_df.columns = ["feature", "variance"]
else:
    top_var_df = pd.DataFrame(columns=["feature", "variance"])

# Check for SYDE1 (including variants like 'SYDE1 (85360)' or 'ENSG...')
syde1_matches = [c for c in df.columns if "SYDE1" in c.upper() or "ENSG" in c.upper()]
if syde1_matches:
    has_syde1 = True
    syde1_note = f"present as {', '.join(syde1_matches[:5])}"  # limit display to first 5 matches
else:
    has_syde1 = False
    syde1_note = "NOT found"
print(f"SYDE1 column: {syde1_note}")

# Write summaries
summary_txt = os.path.join(OUT_DIR, "eda_summary.txt")
print(f"Writing summary to {summary_txt} ...")
with open(summary_txt, "w") as f:
    f.write("=== EDA SUMMARY ===\n")
    f.write(f"Rows (models): {n_models}\n")
    f.write(f"Columns (features): {n_features}\n\n")
    f.write("Missingness >20% (top 50):\n")
    if not high_missing.empty:
        f.write(high_missing.head(50).to_string() + "\n\n")
    else:
        f.write("None\n\n")
    f.write(f"Constant/near-constant columns (count): {len(constant_cols)}\n")
    if len(constant_cols) > 0:
        f.write((",".join(constant_cols[:100])) + ("\n...\n" if len(constant_cols) > 100 else "\n"))
    f.write("\nFeature type counts:\n")
    f.write(f"Binary-like: {len(binary_cols)}\n")
    f.write(f"Continuous: {len(cont_cols)}\n\n")
    f.write(f"SYDE1 column: {syde1_note}\n")

binary_csv = os.path.join(OUT_DIR, "binary_feature_prevalence.csv")
cont_csv = os.path.join(OUT_DIR, "top_variance_expression_features.csv")

if not binary_summary.empty:
    print(f"Writing binary feature prevalence to {binary_csv} ...")
    binary_summary.to_csv(binary_csv, index=False)
else:
    print("No binary-like features detected to write.")

if not top_var_df.empty:
    print(f"Writing top-variance continuous features to {cont_csv} ...")
    top_var_df.to_csv(cont_csv, index=False)
else:
    print("No continuous features detected to write.")

print("EDA Step 1 complete.")