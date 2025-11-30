import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/top_fak_features.csv")

plt.figure(figsize=(8,6))
plt.barh(df["feature"].head(20), df["importance"].head(20))
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 FAK Predictive Biomarkers")
plt.tight_layout()
plt.savefig("outputs/top_fak_features_plot.png", dpi=300)
print("Saved outputs/top_fak_features_plot.png")