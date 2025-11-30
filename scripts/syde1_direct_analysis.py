import pandas as pd

# Load data
features = pd.read_csv("outputs/processed_features.csv")
deps = pd.read_csv("outputs/dependencies.csv")

# --- SYDE1 check ---
syde1_rows = features[features["biomarker"].astype(str).eq("SYDE1 (85360)")]
print(f"SYDE1 rows in processed_features: {len(syde1_rows)}")
if len(syde1_rows) > 0:
    print(syde1_rows.sort_values("importance_score", ascending=False).head(10))

# --- PTK2 check ---
has_ptk2 = any("PTK2" in c for c in deps.columns)
print(f"PTK2 present in dependencies.csv: {has_ptk2}")

# --- dropped SYDE1 rows ---
valid_deps = set(deps.columns)
dropped = syde1_rows[~syde1_rows["dependency"].astype(str).isin(valid_deps)]
print(f"SYDE1 rows dropped by dependency filter: {len(dropped)}")