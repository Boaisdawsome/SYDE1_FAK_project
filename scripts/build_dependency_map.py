import pandas as pd
import os

# Path to DepMap model mapping file
mapping_path = "data/Achilles_model_depmap_mapping.csv"  # adjust if filename differs
out_path = "outputs/dependency_map.csv"

# Load mapping
df = pd.read_csv(mapping_path)

# Some DepMap versions call it 'ModelID' or 'DepMap_ID'
model_col = [c for c in df.columns if c.lower() in ["modelid", "depmap_id", "ach_id"]][0]
gene_col = [c for c in df.columns if "gene" in c.lower()][0]

depmap = df[[model_col, gene_col]].drop_duplicates()
depmap.columns = ["ACH_ID", "Gene"]

os.makedirs("outputs", exist_ok=True)
depmap.to_csv(out_path, index=False)

print(f"[SUCCESS] dependency_map.csv saved to {out_path} ({len(depmap)} entries)")