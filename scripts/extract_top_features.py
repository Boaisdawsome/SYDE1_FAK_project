import joblib, pandas as pd
import numpy as np

# load model and feature names
model = joblib.load("outputs/fak_baseline_model.pkl")
df = pd.read_csv("outputs/training_dataset.csv")

# keep only numeric columns (same preprocessing)
X = df.drop(columns=["FAK_dependency", "FAK_dependency_score"]).select_dtypes(include=["number"])

# get feature importances
importances = model.feature_importances_
idx = np.argsort(importances)[::-1][:20]  # top 20
top_features = pd.DataFrame({
    "feature": X.columns[idx],
    "importance": importances[idx]
})
print(top_features)
top_features.to_csv("outputs/top_fak_features.csv", index=False)
print("\n✅ Saved top features to outputs/top_fak_features.csv")
