import pandas as pd

df = pd.read_csv("data/primary-screen-replicate-collapsed-logfold-change.csv")

# Show all unique drug names
print(sorted(df['name'].dropna().unique())[:200])