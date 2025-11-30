import pandas as pd

df = pd.read_csv("outputs/dependencies.csv")
df["FAK_dependency_score"] = df["PTK2 (5747)"]

# Define FAK-dependent as bottom 10% of PTK2 scores (most dependent cell lines)
threshold = df["FAK_dependency_score"].quantile(0.1)
df["FAK_dependency"] = (df["FAK_dependency_score"] <= threshold).astype(int)

# Save results
df[["ModelID", "FAK_dependency_score", "FAK_dependency"]].to_csv("outputs/fak_target.csv", index=False)
print(f"Threshold used: {threshold:.4f}")
print(df["FAK_dependency"].value_counts())