import pandas as pd
import os
import numpy as np
import scipy.stats as st

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

merged_path = os.path.join(OUT_DIR, "merged_biomarkers.csv")
deps_path = os.path.join(OUT_DIR, "dependencies.csv")
out_path = os.path.join(OUT_DIR, "processed_features.csv")

print(f"Checking for merged biomarker data at: {merged_path}")
if not os.path.exists(merged_path):
    raise FileNotFoundError(f"[ERROR] File does not exist: {merged_path}")

print(f"Loading merged biomarker data from: {merged_path}")
merged = pd.read_csv(merged_path)
print(f"[INFO] Merged DataFrame loaded. Shape: {merged.shape}")
# --- Load dependencies & integrate REAL feature–dependency mapping ---
print(f"Loading dependencies from: {deps_path}")
deps = pd.read_csv(deps_path)

# Fix common index-name issue
if "ModelID" not in merged.columns and "Unnamed: 0" in merged.columns:
    print("[WARN] 'ModelID' missing in merged_biomarkers.csv; renaming 'Unnamed: 0' -> 'ModelID'")
    merged = merged.rename(columns={"Unnamed: 0": "ModelID"})
if "ModelID" not in deps.columns and "Unnamed: 0" in deps.columns:
    print("[WARN] 'ModelID' missing in dependencies.csv; renaming 'Unnamed: 0' -> 'ModelID'")
    deps = deps.rename(columns={"Unnamed: 0": "ModelID"})

if "ModelID" not in merged.columns or "ModelID" not in deps.columns:
    raise ValueError("Missing 'ModelID' column in either merged_biomarkers.csv or dependencies.csv")

print(f"[INFO] Dependencies DataFrame loaded. Shape: {deps.shape}")
print(f"[INFO] Dependency columns (sample): {deps.columns[:8].tolist()}")

# Prep matrices aligned on the same models
bio = merged.set_index("ModelID").apply(pd.to_numeric, errors="coerce")
dep = deps.set_index("ModelID").apply(pd.to_numeric, errors="coerce")

common_models = bio.index.intersection(dep.index)
if len(common_models) == 0:
    raise ValueError("No overlapping ModelID rows between biomarkers and dependencies.")
bio = bio.loc[common_models].dropna(axis=1, how="all")
dep = dep.loc[common_models].dropna(axis=1, how="all")

print(f"[INFO] Overlap models: {len(common_models)}")
print(f"[INFO] Biomarker matrix: {bio.shape}")
print(f"[INFO] Dependency matrix: {dep.shape}")

# Z-score per column
def zscore(df):
    m = df.mean(axis=0)
    s = df.std(axis=0, ddof=0).replace(0, pd.NA)
    return ((df - m) / s).fillna(0.0)

bio_z = zscore(bio)
dep_z = zscore(dep)
# Align biomarker and dependency matrices to ensure identical ModelID ordering
bio_z, dep_z = bio_z.align(dep_z, join="inner", axis=0)
print(f"[DEBUG] Aligned rows: {bio_z.shape[0]} vs {dep_z.shape[0]}")

# Correlations via block dot-product
n = float(len(common_models))
TOPK_PER_BIOMARKER = 50
MIN_ABS_CORR = 0.15

biomarker_names = bio_z.columns.tolist()
rows_out = []
BATCH = 200

for i in range(0, len(biomarker_names), BATCH):
    cols = biomarker_names[i:i+BATCH]
    corr_block = (bio_z[cols].T @ dep_z) / n  # (batch x deps)
    for b in cols:
        s = corr_block.loc[b].abs()
        top = s.nlargest(TOPK_PER_BIOMARKER)
        top = top[top >= MIN_ABS_CORR]
        for dep_gene, score in top.items():
            rows_out.append((b, dep_gene, float(score)))

print(f"[INFO] Correlation-derived links: {len(rows_out):,}")

import pandas as pd
out_df = pd.DataFrame(rows_out, columns=["biomarker", "dependency", "importance_score"])
if not out_df.empty:
    out_df["importance_score"] = out_df.groupby("dependency")["importance_score"].transform(
        lambda x: x / (x.max() if x.max() > 0 else 1.0)
    )

print(f"[INFO] Final processed features shape: {out_df.shape}")
out_df.to_csv(out_path, index=False)
print(f"[SUCCESS] Processed features saved to: {out_path}")

# --- Direct SYDE1↔PTK2 correlation test ---
try:
    print("\n[INFO] Running SYDE1↔PTK2 correlation validation...")

    # Reload the aligned biomarker and dependency matrices
    sy_expr = bio.get("SYDE1 (85360)")
    if sy_expr is None:
        print("[WARN] SYDE1 column not found in biomarkers.")
    else:
        ptk2_col = [c for c in dep.columns if "PTK2" in c.upper()]
        if len(ptk2_col) == 0:
            print("[WARN] PTK2 not found in dependency dataset.")
        else:
            ptk2_dep = dep[ptk2_col[0]]
            mask = sy_expr.notna() & ptk2_dep.notna()
            if mask.sum() < 5:
                print(f"[WARN] Insufficient overlapping models ({mask.sum()}) for correlation.")
            else:
                r, p = st.pearsonr(sy_expr[mask], ptk2_dep[mask])
                rho, pp = st.spearmanr(sy_expr[mask], ptk2_dep[mask])
                print(f"[RESULT] SYDE1↔PTK2 Pearson r={r:.3f}, p={p:.3g} | Spearman rho={rho:.3f}, p={pp:.3g}")
except Exception as e:
    print(f"[ERROR] SYDE1↔PTK2 correlation test failed: {e}")

# Quick SYDE1↔PTK2 check (informational)
try:
    sy = out_df[out_df["biomarker"].str.contains("SYDE1", case=False, na=False)]
    sp = sy[sy["dependency"].str.contains("PTK2", case=False, na=False)]
    print(f"[CHECK] SYDE1 links: {len(sy)} ; SYDE1↔PTK2: {len(sp)}")
    if not sp.empty:
        print(sp.sort_values("importance_score", ascending=False).head(5))
except Exception as e:
    print(f"[WARN] SYDE1/PTK2 check skipped: {e}")