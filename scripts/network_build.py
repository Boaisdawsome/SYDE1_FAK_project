import pandas as pd
import networkx as nx
import os

#setups and paths
OUT_DIR = "outputs"
features_path = os.path.join(OUT_DIR, "processed_features.csv")
deps_path = os.path.join(OUT_DIR, "dependencies.csv")

#load data
features_df = pd.read_csv(features_path)
dependencies_df = pd.read_csv(deps_path)

# Map ACH IDs to gene symbols if mapping file exists and is valid
dep_map_path = os.path.join(OUT_DIR, "dependency_map.csv")
try:
    dep_map_df = pd.read_csv(dep_map_path)
    if "ACH_ID" in dep_map_df.columns and "Gene" in dep_map_df.columns:
        achid_to_gene = dict(zip(dep_map_df["ACH_ID"].astype(str), dep_map_df["Gene"].astype(str)))
        features_df["dependency"] = features_df["dependency"].astype(str).map(achid_to_gene).fillna(features_df["dependency"])
        print("[INFO] Mapped ACH IDs in 'dependency' column to gene symbols using dependency_map.csv")
    else:
        print("[WARN] 'dependency_map.csv' missing required columns ('ACH_ID', 'Gene'). No mapping applied.")
except FileNotFoundError:
    print("[WARN] 'dependency_map.csv' not found. No mapping applied.")

# Check and extract dependency column
print(f"[DEBUG] Dependency file columns: {dependencies_df.columns.tolist()}")
if "dependency" in dependencies_df.columns:
    dependencies = dependencies_df["dependency"].astype(str).unique()
else:
    first_col = dependencies_df.columns[0]
    print(f"[WARN] No 'dependency' column found. Using first column '{first_col}' instead.")
    dependencies = dependencies_df[first_col].astype(str).unique()

# Initialize bipartite graph
B = nx.Graph()

# Add biomarker nodes
biomarkers = features_df["biomarker"].unique()
B.add_nodes_from(biomarkers, bipartite="biomarker")

# Add dependency nodes
B.add_nodes_from(dependencies, bipartite="dependency")

# Vectorized edge creation — prevents CPU overload
print("[INFO] Building edges (fast mode)...")

# Limit to top ~60k edges by importance_score for balance between coverage and performance
# Use top 500,000 edges for broader connectivity; keep moderately strong edges for under-connected biomarkers like SYDE1
features_df = features_df.sort_values(by="importance_score", ascending=False).head(500000)
print(f"[INFO] Using top {len(features_df):,} edges by importance_score.")
edges = list(zip(features_df["biomarker"], features_df["dependency"], features_df["importance_score"]))
B.add_weighted_edges_from(edges)
print(f"[INFO] Added {len(edges):,} edges.")

# Output summary
print(f"Number of biomarker nodes: {len(biomarkers)}")
print(f"Number of dependency nodes: {len(dependencies)}")
print(f"Number of edges: {B.number_of_edges()}")

# Save network
import pickle
with open(os.path.join(OUT_DIR, "bipartite_network.gpickle"), "wb") as f:
    pickle.dump(B, f, protocol=pickle.HIGHEST_PROTOCOL)
print("[SUCCESS] Bipartite network saved to outputs/bipartite_network.gpickle")

# Louvain community detection
import community
print("[INFO] Running Louvain community detection...")
partition = community.best_partition(B, resolution=1.0)
modularity_score = community.modularity(partition, B)
print(f"[INFO] Louvain modularity score: {modularity_score:.4f}")

# Save community assignments
partition_df = pd.DataFrame(list(partition.items()), columns=["node", "community"])
partition_csv_path = os.path.join(OUT_DIR, "community_assignments.csv")
partition_df.to_csv(partition_csv_path, index=False)
print(f"[SUCCESS] Community assignments saved to {partition_csv_path}")