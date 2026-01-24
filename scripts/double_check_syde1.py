import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Config
# -----------------------------
EXPR_PATH = "data/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
DEPS_PATH = "data/CRISPRGeneDependency.csv"
TOPFEAT_PATH = "outputs/top_fak_features.csv"   # optional
OUT_DIR = "outputs"

SYDE1_SYMBOL = "SYDE1"
PTK2_SYMBOL = "PTK2"
BOTTOM_Q = 0.10   # bottom 10% PTK2 Chronos = FAK-dependent

RANDOM_STATE = 42

def find_gene_col(cols, symbol):
    # matches "SYDE1 (85360)" etc.
    symbol = symbol.strip().upper()
    for c in cols:
        if str(c).upper().startswith(symbol + " "):
            return c
    # fallback: exact match
    for c in cols:
        if str(c).strip().upper() == symbol:
            return c
    return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[LOAD] Reading expression + dependency files...")
    expr = pd.read_csv(EXPR_PATH)
    deps = pd.read_csv(DEPS_PATH)

    # Expression has ModelID column already; keep just ModelID + gene cols later
    if "ModelID" not in expr.columns:
        raise RuntimeError("Expression file missing ModelID column (unexpected).")

    # Dependencies: ModelIDs live in 'Unnamed: 0' in your DepMap dump
    if "ModelID" not in deps.columns:
        if "Unnamed: 0" in deps.columns:
            deps = deps.rename(columns={"Unnamed: 0": "ModelID"})
        else:
            raise RuntimeError("Dependency file missing both ModelID and Unnamed: 0.")

    # Detect actual gene columns
    syde1_col_expr = find_gene_col(expr.columns, SYDE1_SYMBOL)
    ptk2_col_deps = find_gene_col(deps.columns, PTK2_SYMBOL)

    print(f"[COLS] SYDE1 expr col: {syde1_col_expr}")
    print(f"[COLS] PTK2 dep col:   {ptk2_col_deps}")

    if syde1_col_expr is None:
        raise RuntimeError("Could not find SYDE1 column in expression file.")
    if ptk2_col_deps is None:
        raise RuntimeError("Could not find PTK2 column in dependency file.")

    # Keep only needed cols, drop dupes safely
    expr_small = expr[["ModelID", syde1_col_expr]].copy()
    # Some ModelIDs appear multiple times in expression: collapse by mean
    before = expr_small.shape[0]
    expr_small = expr_small.groupby("ModelID", as_index=False)[syde1_col_expr].mean()
    after = expr_small.shape[0]
    if after != before:
        print(f"[FIX] Collapsed expr duplicates: {before} -> {after}")

    deps_small = deps[["ModelID", ptk2_col_deps]].copy()
    deps_small = deps_small.drop_duplicates(subset=["ModelID"])

    # Merge on ModelID
    df = expr_small.merge(deps_small, on="ModelID", how="inner")
    df = df.rename(columns={syde1_col_expr: "SYDE1_expr", ptk2_col_deps: "PTK2_dep"})
    df = df.dropna(subset=["SYDE1_expr", "PTK2_dep"])

    print(f"[MERGE] N overlap after cleaning: {df.shape[0]}")
    if df.shape[0] < 50:
        raise RuntimeError("Too few overlapping rows — something still off.")

    # Define FAK-dependent: bottom 10% PTK2 Chronos (lower = more dependent)
    thr = df["PTK2_dep"].quantile(BOTTOM_Q)
    df["FAK_dependent"] = (df["PTK2_dep"] <= thr).astype(int)

    print(f"[TARGET] Bottom {int(BOTTOM_Q*100)}% threshold (PTK2 Chronos) = {thr:.4f}")
    print(df["FAK_dependent"].value_counts().rename("count"))

    # -----------------------------------------
    # CHECK 1: Direction + effect size
    # -----------------------------------------
    m0 = df.loc[df["FAK_dependent"] == 0, "SYDE1_expr"].mean()
    m1 = df.loc[df["FAK_dependent"] == 1, "SYDE1_expr"].mean()
    print("\n[CHECK 1] Mean SYDE1 expression:")
    print(f"  Not FAK-dependent (0): {m0:.4f}")
    print(f"  FAK-dependent (1):     {m1:.4f}")
    print("  Interpretation: if mean(1) < mean(0), then LOW SYDE1 -> MORE FAK dependency.")

    # -----------------------------------------
    # CHECK 2: SYDE1-only predictability (AUC)
    # -----------------------------------------
    X = df[["SYDE1_expr"]].values
    y = df["FAK_dependent"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr, ytr)
    auc_lr = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])

    rf1 = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    rf1.fit(Xtr, ytr)
    auc_rf1 = roc_auc_score(yte, rf1.predict_proba(Xte)[:, 1])

    print("\n[CHECK 2] SYDE1-only predictability:")
    print(f"  LogisticRegression AUC: {auc_lr:.4f}")
    print(f"  RandomForest (SYDE1-only) AUC: {auc_rf1:.4f}")

    # Save the clean merged df for plotting / paper
    out_df_path = os.path.join(OUT_DIR, "syde1_ptk2_merged.csv")
    df.to_csv(out_df_path, index=False)
    print(f"\n[SAVED] {out_df_path}")

    # -----------------------------------------
    # CHECK 3: Ablation test on your top RF features (if available)
    # -----------------------------------------
    if os.path.exists(TOPFEAT_PATH):
        print("\n[CHECK 3] Ablation test using outputs/top_fak_features.csv ...")

        top = pd.read_csv(TOPFEAT_PATH)
        if "feature" not in top.columns:
            raise RuntimeError("top_fak_features.csv missing 'feature' column.")

        top_feats = top["feature"].astype(str).tolist()

        # Build reduced feature matrix from merged_biomarkers or training_dataset if present
        # Prefer training_dataset (already aligned + cleaned in your pipeline)
        reduced_source = None
        if os.path.exists("outputs/training_dataset.csv"):
            reduced_source = "outputs/training_dataset.csv"
            full = pd.read_csv(reduced_source)
            # Expect ModelID exists and FAK_dependency exists in training_dataset
            if "ModelID" not in full.columns:
                raise RuntimeError("training_dataset.csv missing ModelID.")
            if "FAK_dependency" not in full.columns:
                raise RuntimeError("training_dataset.csv missing FAK_dependency.")

            # Keep rows that exist in df (same ModelIDs)
            full = full[full["ModelID"].isin(df["ModelID"])].copy()
            full = full.set_index("ModelID")

            # Ensure features exist
            use_feats = [f for f in top_feats if f in full.columns]
            # Force-add SYDE1 expr feature if present in this file
            # (often it's like "SYDE1 (85360)")
            syde1_any = [c for c in full.columns if str(c).startswith("SYDE1 ")]
            if syde1_any and syde1_any[0] not in use_feats:
                use_feats.append(syde1_any[0])

            if len(use_feats) < 5:
                print("  [SKIP] Too few top features found in training_dataset.csv")
                return

            Y = full["FAK_dependency"].values
            X_full = full[use_feats].select_dtypes(include=["number"]).fillna(0.0)

            # split
            Xtr, Xte, ytr, yte = train_test_split(
                X_full.values, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y
            )

            rf = RandomForestClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            )
            rf.fit(Xtr, ytr)
            auc_with = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])

            # remove SYDE1 if present
            syde_cols = [c for c in use_feats if str(c).startswith("SYDE1 ")]
            if syde_cols:
                use_feats_wo = [c for c in use_feats if c not in syde_cols]
                X_wo = full[use_feats_wo].select_dtypes(include=["number"]).fillna(0.0).values
                Xtr, Xte, ytr, yte = train_test_split(
                    X_wo, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y
                )
                rf2 = RandomForestClassifier(
                    n_estimators=600,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1
                )
                rf2.fit(Xtr, ytr)
                auc_without = roc_auc_score(yte, rf2.predict_proba(Xte)[:, 1])

                print(f"  AUC with SYDE1:    {auc_with:.4f}")
                print(f"  AUC without SYDE1: {auc_without:.4f}")
                print(f"  Drop (with - without): {(auc_with - auc_without):.4f}")
            else:
                print("  [NOTE] No SYDE1 column found in training_dataset.csv, ablation skipped.")
        else:
            print("  [SKIP] outputs/training_dataset.csv not found, ablation skipped.")
    else:
        print("\n[CHECK 3] outputs/top_fak_features.csv not found — skipping ablation test.")

    print("\n[DONE] Double-check complete.")

if __name__ == "__main__":
    main()