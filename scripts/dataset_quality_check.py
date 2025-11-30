import pandas as pd

# Load datasets
dep = pd.read_csv("CRISPR_Dependency.csv")
expr = pd.read_csv("Expression_Public.csv")
mut = pd.read_csv("Mutations_Public.csv")
copy = pd.read_csv("CopyNumber_Public.csv")

# Inspect structure
print("Dependency:", dep.shape)
print("Expression:", expr.shape)
print("Mutation:", mut.shape)
print("Copy Number:", copy.shape)

# Quick look
print(dep.head())
print(expr.head())

# Missing value check
print(dep.isna().sum().sum())
