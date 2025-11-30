import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

print("[INFO] Loading dataset...")
df = pd.read_csv("outputs/training_dataset.csv")
print(f"[INFO] Loaded shape: {df.shape}")

# Split
X = df.drop(columns=["FAK_dependency", "FAK_dependency_score"])
# remove all non-numeric columns (like IDs or gene names)
X = X.select_dtypes(include=["number"])
y = df["FAK_dependency"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
print("[INFO] Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("[RESULTS] ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))