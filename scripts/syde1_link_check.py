import pandas as pd
model = pd.read_csv("data/Model.csv")
model_cols = model.columns
print(model_cols[:10])  # find the right column names