import pandas as pd

features = pd.read_csv("outputs/processed_features.csv")
target = pd.read_csv("outputs/fak_target.csv")

merged = pd.merge(features, target, on="ModelID")
print("Merged shape:", merged.shape)
print(merged.head(5))
