import os
import pandas as pd
import numpy as np
import time

"""
Preprocess DepMap-style omics where each table ALREADY contains a ModelID column.

Key fixes vs prior version:
- Stop trying to align via DepMap_ID / ProfileID when ModelID is present.
- Standardize index on ModelID, drop metadata, coerce to numeric.
- Binarize mutation & CNV correctly (with CNV threshold) after numeric coercion.
- Robust merging with suffixing only when true name collisions happen.
- Clear diagnostics: model counts per table, overlap sizes, final shape.
"""

print("Starting preprocessing...")
start_time = time.time()

# ---------------------------
# Paths
# ---------------------------
# Define input/output directories and ensure output directory exists
DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "merged_biomarkers.csv")

# ---------------------------
# Helpers
# ---------------------------
# Define metadata columns to drop from datasets to keep only relevant features
META_COLS_CANDIDATES = {
    "common": {
        "Unnamed: 0", "SequencingID", "IsDefaultEntryForModel", "ModelConditionID",
        "ProfileID", "DataType", "Stranded", "SequencingDate", "DepMapCode",
        "Lineage", "CellFormat", "GrowthMedia", "GrowthPattern", "SourceModelCondition"
    }
}

def drop_metadata_cols(df: pd.DataFrame, keep: set = None) -> pd.DataFrame:
    """Drop obvious metadata columns if present (except any in `keep`)."""
    if keep is None:
        keep = set()
    drop_cols = [c for c in df.columns if c in META_COLS_CANDIDATES["common"] and c not in keep]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df



def set_index_on_modelid(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Ensure ModelID is the index. If both index and column include ModelID, prefer the column."""
    if "ModelID" in df.columns:
        df = df.set_index("ModelID", drop=True)
    else:
        # Some files may already have ModelID as index name
        if df.index.name != "ModelID":
            # If they don't, give up loudly so we don't silently misalign
            raise ValueError(f"[{dataset_name}] No 'ModelID' column or index found.")
    return df

def coerce_numeric_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Coerce all columns to numeric where possible; non-numeric columns become NaN."""
    # Preserve index, act on columns only
    numeric = df.apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN after coercion (pure text cols that slipped through)
    all_nan_cols = numeric.columns[numeric.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        print(f"[{dataset_name}] Dropping {len(all_nan_cols)} all-NaN columns after numeric coercion.")
        numeric = numeric.drop(columns=list(all_nan_cols))
    return numeric

def binarize_any_nonzero(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Turn any non-zero numeric entry into 1; NaNs become 0."""
    # Fill NaN with 0 first, then >0 -> 1
    df = df.fillna(0)
    # Compare absolute value to tolerate small numeric noise
    return (df.abs() > 0).astype(np.int8)

def binarize_cnv_loss(df: pd.DataFrame, threshold: float, dataset_name: str) -> pd.DataFrame:
    """
    CNV loss binarization: 1 if value < threshold (e.g., -0.3), else 0. NaNs => 0.
    """
    df = df.fillna(0)
    return (df < threshold).astype(np.int8)

def load_omics_table(path: str, dataset_name: str) -> pd.DataFrame:
    """Generic loader that trusts ModelID is a column in wide format."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = drop_metadata_cols(df, keep={"ModelID"})  # keep ModelID if present
    df = set_index_on_modelid(df, dataset_name)
    return df

def suffix_overlaps(left: pd.DataFrame, right: pd.DataFrame, right_suffix: str) -> pd.DataFrame:
    """If left and right share any column names, suffix the right to avoid collisions."""
    overlap = set(left.columns).intersection(set(right.columns))
    if overlap:
        right = right.rename(columns={c: f"{c}_{right_suffix}" for c in overlap})
    return right

def report_overlap(label_a: str, idx_a, label_b: str, idx_b):
    """Print overlap counts for transparency."""
    inter = len(set(idx_a).intersection(set(idx_b)))
    print(f"[Overlap] {label_a} ∩ {label_b} = {inter}")

# ---------------------------
# Load OmicsProfiles (optional for sanity check, not used for merge)
# ---------------------------
# Attempt to load mapping between ProfileID and ModelID for reference
profiles_path = os.path.join(DATA_DIR, "OmicsProfiles.csv")
profiles = None
if os.path.exists(profiles_path):
    try:
        profiles = pd.read_csv(profiles_path, usecols=["ProfileID", "ModelID"])
        profiles = profiles.drop_duplicates()
        print(f"Loaded OmicsProfiles: {profiles.shape[0]} mappings (ProfileID→ModelID).")
    except Exception:
        print("Loaded OmicsProfiles (columns may differ); not used for merge because datasets already have ModelID.")

# ---------------------------
# Load datasets that YOU actually have
# ---------------------------
# Load expression, mutation, and CNV datasets; preprocess and binarize as needed
print("Loading Expression...")
expr = load_omics_table(
    os.path.join(DATA_DIR, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"),
    "Expression"
)
if expr is not None:
    # Expression is already log2(TPM+1). Just coerce to numeric in case.
    expr = coerce_numeric_frame(expr, "Expression")
    print(f"[Expression] models={expr.shape[0]}, features={expr.shape[1]}")
print(f"Elapsed time after loading Expression: {time.time() - start_time:.2f} seconds")

print("Loading Mutation (damaging)...")
mut_dmg = load_omics_table(
    os.path.join(DATA_DIR, "OmicsSomaticMutationsMatrixDamaging.csv"),
    "MutationDamaging"
)
if mut_dmg is not None:
    mut_dmg = coerce_numeric_frame(mut_dmg, "MutationDamaging")
    mut_dmg = binarize_any_nonzero(mut_dmg, "MutationDamaging")
    print(f"[MutationDamaging] models={mut_dmg.shape[0]}, features={mut_dmg.shape[1]}")
print(f"Elapsed time after loading MutationDamaging: {time.time() - start_time:.2f} seconds")

print("Loading Mutation (hotspot)...")
mut_hot = load_omics_table(
    os.path.join(DATA_DIR, "OmicsSomaticMutationsMatrixHotspot.csv"),
    "MutationHotspot"
)
if mut_hot is not None:
    mut_hot = coerce_numeric_frame(mut_hot, "MutationHotspot")
    mut_hot = binarize_any_nonzero(mut_hot, "MutationHotspot")
    print(f"[MutationHotspot] models={mut_hot.shape[0]}, features={mut_hot.shape[1]}")
print(f"Elapsed time after loading MutationHotspot: {time.time() - start_time:.2f} seconds")

print("Loading CNV (WGS)...")
cnv = load_omics_table(
    os.path.join(DATA_DIR, "OmicsCNGeneWGS.csv"),
    "CNV_WGS"
)
if cnv is not None:
    cnv = coerce_numeric_frame(cnv, "CNV_WGS")
    cnv = binarize_cnv_loss(cnv, threshold=-0.3, dataset_name="CNV_WGS")
    print(f"[CNV_WGS] models={cnv.shape[0]}, features={cnv.shape[1]}")
print(f"Elapsed time after loading CNV_WGS: {time.time() - start_time:.2f} seconds")

# Collect only the datasets that actually loaded
tables = []
labels = []

if expr is not None and expr.shape[0] > 0:
    tables.append(("expr", expr))
    labels.append("expr")
if mut_dmg is not None and mut_dmg.shape[0] > 0:
    tables.append(("mut_dmg", mut_dmg))
    labels.append("mut_dmg")
if mut_hot is not None and mut_hot.shape[0] > 0:
    tables.append(("mut_hot", mut_hot))
    labels.append("mut_hot")
if cnv is not None and cnv.shape[0] > 0:
    tables.append(("cnv", cnv))
    labels.append("cnv")

if not tables:
    raise RuntimeError("No usable datasets were loaded. Check file paths and formats.")

# ---------------------------
# Overlap diagnostics BEFORE merge
# ---------------------------
# Report the number of shared models (ModelID) between each pair of datasets
for i in range(len(tables)):
    for j in range(i + 1, len(tables)):
        (la, A), (lb, B) = tables[i], tables[j]
        report_overlap(la, A.index, lb, B.index)

# ---------------------------
# Merge (inner joins across ModelID)
# ---------------------------
# Merge all datasets by inner join on ModelID, suffixing columns to avoid name collisions
print("Merging all biomarker dataframes (inner join on ModelID)...")
merge_start = time.time()

merged = None
for name, df in tables:
    if merged is None:
        merged = df.copy()
    else:
        # Suffix collisions to avoid "columns overlap but no suffix specified" errors
        df_suffixed = suffix_overlaps(merged, df, right_suffix=name)
        merged = merged.join(df_suffixed, how="inner")

merge_elapsed = time.time() - merge_start
print(f"Merging completed in {merge_elapsed:.2f} seconds")
if merge_elapsed > 30:
    print("Warning: Merging took more than 30 seconds. Consider saving intermediate results as .parquet for faster processing.")

if merged is None or merged.shape[0] == 0:
    # Provide additional hints
    msg = [
        "Merged result is empty (0 rows). Likely no shared ModelID overlap across datasets.",
        "Quick checks:",
        "- Confirm each file truly uses ModelID values like ACH-000xxx.",
        "- Confirm there aren’t duplicate ModelID rows per table (should be unique).",
        "- If some tables carry per-ProfileID rows, pivot first before merging."
    ]
    raise RuntimeError("\n".join(msg))

# ModelID back to a column for CSV clarity
merged = merged.reset_index().rename(columns={"index": "ModelID"})

print(f"Final merged shape: models={merged.shape[0]}, features={merged.shape[1]-1} (excluding ModelID)")

# ---------------------------
# Save
# ---------------------------
# Save the merged dataframe to CSV file
save_start = time.time()
merged.to_csv(OUT_FILE, index=False)
save_elapsed = time.time() - save_start
print(f"Saved merged biomarkers to {OUT_FILE} in {save_elapsed:.2f} seconds")
if save_elapsed > 30:
    print("Warning: Saving merged_biomarkers.csv took more than 30 seconds. Consider saving as .parquet for better performance.")

print("Preprocessing complete.")

# ---------------------------
# Additional Outputs for Network Build
# ---------------------------
print("Saving processed features and dependencies for network build...")

# Save processed features (merged biomarkers)
processed_features_path = os.path.join(OUT_DIR, "processed_features.csv")
save_pf_start = time.time()
merged.to_csv(processed_features_path, index=False)
save_pf_elapsed = time.time() - save_pf_start
print(f"Saved processed features to {processed_features_path} in {save_pf_elapsed:.2f} seconds")
if save_pf_elapsed > 30:
    print("Warning: Saving processed_features.csv took more than 30 seconds. Consider saving as .parquet for better performance.")

# Load dependencies (CRISPRGeneDependency.csv)
dep_path = os.path.join(DATA_DIR, "CRISPRGeneDependency.csv")
if os.path.exists(dep_path):
    deps = pd.read_csv(dep_path)
    if "ModelID" in deps.columns:
        deps = deps.set_index("ModelID")
    else:
        print("Warning: 'ModelID' column not found in dependency data; using default index.")
    deps_out_path = os.path.join(OUT_DIR, "dependencies.csv")
    save_dep_start = time.time()
    deps.to_csv(deps_out_path)
    save_dep_elapsed = time.time() - save_dep_start
    print(f"Saved dependencies to {deps_out_path} in {save_dep_elapsed:.2f} seconds")
    if save_dep_elapsed > 30:
        print("Warning: Saving dependencies.csv took more than 30 seconds. Consider saving as .parquet for better performance.")
else:
    print("Warning: CRISPRGeneDependency.csv not found in data directory.")

total_elapsed = time.time() - start_time
print(f"Total preprocessing runtime: {total_elapsed:.2f} seconds")