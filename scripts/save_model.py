import joblib
from sklearn.ensemble import RandomForestClassifier

# load your trained model if running from same session isn’t possible
# or if you’re continuing later, retrain by importing from train_baseline_model

import pandas as pd
df = pd.read_csv("outputs/training_dataset.csv")

X = df.drop(columns=["FAK_dependency", "FAK_dependency_score"])
X = X.select_dtypes(include=["number"])
y = df["FAK_dependency"]

model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X, y)

joblib.dump(model, "outputs/fak_baseline_model.pkl")
print("✅ Model saved to outputs/fak_baseline_model.pkl")