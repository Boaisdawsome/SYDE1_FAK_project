import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

crispr_path = os.path.join(BASE_DIR, "data", "CRISPRGeneEffect.csv")
rna_path = os.path.join(BASE_DIR, "data", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

crispr_df = pd.read_csv(crispr_path)
rna_df = pd.read_csv(rna_path)

print("\nCRISPR HEAD:")
print(crispr_df.head())

print("\nCRISPR COLUMNS:")
print(crispr_df.columns[:10])

print("\nRNA HEAD:")
print(rna_df.head())

print("\nRNA COLUMNS:")
print(rna_df.columns[:10])