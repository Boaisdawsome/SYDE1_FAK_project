import pandas as pd

# Load treatment info
info = pd.read_csv("data/primary-screen-replicate-collapsed-treatment-info.csv")
# Load LFC drug sensitivity matrix
lfc = pd.read_csv("data/primary-screen-replicate-collapsed-logfold-change.csv")

# True FAK inhibitor names
fak_names = ["VS-4718", "defactinib", "PF-573228", "PF-562271", "NVP-TAE226"]

# Find column_name strings for these drugs
mask = False
for n in fak_names:
    mask |= info["name"].str.contains(n, case=False, na=False)

hits = info[mask]

print("FAK inhibitors found:")
print(hits[["name","broad_id","column_name"]])

# Extract their column_name values
cols = hits["column_name"].tolist()

# Keep only those columns + cell line index
subset = lfc[["Unnamed: 0"] + cols]

subset = subset.rename(columns={"Unnamed: 0": "DepMap_ID"})
subset.to_csv("outputs/fak_lfc_matrix.csv", index=False)

print("\nSaved -> outputs/fak_lfc_matrix.csv")
print(subset.head())