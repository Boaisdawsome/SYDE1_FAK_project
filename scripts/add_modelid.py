import pandas as pd

# load with first column as index
df = pd.read_csv("outputs/merged_biomarkers.csv", index_col=0)

# turn that index into a proper column
df.reset_index(inplace=True)
df.rename(columns={"index": "ModelID"}, inplace=True)

# save new version
df.to_csv("outputs/merged_biomarkers_with_id.csv", index=False)
print("✅ added ModelID column, shape:", df.shape)