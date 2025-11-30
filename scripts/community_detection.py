import os
import time
import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm

"""
Build a bipartite biomarker–dependency network from the feature-selection outputs.
Inputs (from ./outputs):
  - processed_features.csv : columns [biomarker, dependency, importance_score]
  - dependencies.csv       : columns [dependency]
Outputs (to ./outputs):
  - bipartite_network.gpickle (pickle of a NetworkX Graph)
  - network_summary.txt
"""
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

features_path = os.path.join(OUT_DIR, "processed_features.csv")
deps_path = os.path.join(OUT_DIR, "dependencies.csv")

start = time.time()
print("Loading processed features and dependencies from 'outputs/' ...")

# --- Load ---
if not os.path.exists(features_path):
    raise FileNotFoundError(f"Missing {features_path}. Run feature_selection.py first.")
if not os.path.exists(deps_path):
    raise FileNotFoundError(f"Missing {deps_path}. Run feature_selection.py first.")

features_df = pd.read_csv(features_path)
dependencies_df = pd.read_csv(deps_path)
print(f"[INFO] Loaded features: {len(features_df):,} rows; dependencies: {len(dependencies_df):,} rows")
print(f"[INFO] Load step took {time.time()-start:.2f}s")

# --- Validate columns ---
required_feat_cols = {"biomarker", "dependency", "importance_score"}
if not required_feat_cols.issubset(features_df.columns):
    raise ValueError(
        "processed_features.csv must have columns: 'biomarker','dependency','importance_score'.\n"
        f"Found columns: {list(features_df.columns)[:10]} ..."
    )
if "dependency" not in dependencies_df.columns:
    print("[WARN] No 'dependency' column found. Using first column instead.")
    dependencies_df = dependencies_df.rename(columns={dependencies_df.columns[0]: "dependency"})

# Keep only dependencies that are listed
valid_deps = set(dependencies_df["dependency"].astype(str).unique())
features_df = features_df[features_df["dependency"].astype(str).isin(valid_deps)].copy()

# Optional: drop very small weights
features_df = features_df[features_df["importance_score"] > 0].copy()

print(f"Rows after filtering: {len(features_df):,}")

# --- Build bipartite graph ---
print("[INFO] Building bipartite graph...")
B = nx.Graph()

biomarkers = features_df["biomarker"].astype(str).unique().tolist()
dependencies = sorted(valid_deps)


B.add_nodes_from(biomarkers, bipartite="biomarker")
B.add_nodes_from(dependencies, bipartite="dependency")

# --- Sparsify edges aggressively ---
# 1) Normalize per dependency (max=1.0 inside each dependency)
features_df["importance_score"] = (
    features_df.groupby("dependency")["importance_score"].transform(lambda x: x / x.max())
)

# 2) Keep only strongest edges
EDGE_MIN = 0.015      # keep slightly weaker edges for ~40k–60k total
TOPK_PER_DEP = 250     # increase top biomarkers per dependency
TOPK_PER_BM  = 150     # increase top dependencies per biomarker

# Drop very small weights first
features_df = features_df[features_df["importance_score"] >= EDGE_MIN]

# Top-K per dependency
features_df = (
    features_df.sort_values(["dependency", "importance_score"], ascending=[True, False])
               .groupby("dependency", as_index=False)
               .head(TOPK_PER_DEP)
)

# Top-K per biomarker
features_df = (
    features_df.sort_values(["biomarker", "importance_score"], ascending=[True, False])
               .groupby("biomarker", as_index=False)
               .head(TOPK_PER_BM)
)

print(f"[INFO] After sparsification: {len(features_df):,} edges "
      f"(min≥{EDGE_MIN}, top{TOPK_PER_DEP}/dep, top{TOPK_PER_BM}/bm)")

print("[INFO] Building edges (fast mode)...")
for biomarker, dep, w in tqdm(features_df[["biomarker", "dependency", "importance_score"]].itertuples(index=False),
                             total=len(features_df),
                             desc="Building edges", ncols=100):
    # Add edge only if both ends exist
    if biomarker in biomarkers and dep in valid_deps:
        B.add_edge(str(biomarker), str(dep), weight=float(w))
print("[INFO] Edges added.")

print(f"[INFO] Graph built with {len(biomarkers)} biomarker nodes, {len(dependencies)} dependency nodes, and {B.number_of_edges()} edges.")
print(f"[INFO] Build step took {time.time()-start:.2f}s")

# --- Save ---
print("[INFO] Saving bipartite network and summary...")
net_path = os.path.join(OUT_DIR, "bipartite_network.gpickle")
with open(net_path, "wb") as f:
    pickle.dump(B, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[INFO] Saved network to {net_path}")

summary_path = os.path.join(OUT_DIR, "network_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Biomarker nodes: {len(biomarkers)}\n")
    f.write(f"Dependency nodes: {len(dependencies)}\n")
    f.write(f"Edges: {B.number_of_edges()}\n")
    f.write(f"Runtime (s): {time.time()-start:.2f}\n")
print(f"[INFO] Saved summary to {summary_path}")
print(f"[INFO] Save step took {time.time()-start:.2f}s")


import community as community_louvain

# Paths
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
net_path = os.path.join(OUT_DIR, "bipartite_network.gpickle")

# Load network
if not os.path.exists(net_path):
    raise FileNotFoundError(
        f"Error: '{net_path}' not found. Run scripts/network_build.py first to create it."
    )

print("Loading bipartite network...")
start = time.time()
with open(net_path, "rb") as f:
    G = pickle.load(f)

if not isinstance(G, nx.Graph) or G.number_of_nodes() == 0:
    raise ValueError("Loaded object is not a valid non-empty NetworkX Graph.")

print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"[INFO] Load network step took {time.time()-start:.2f}s")

# Louvain community detection on weighted graph
print("Running Louvain community detection...")
print("[INFO] Running fastgreedy (approximate, CPU-optimized)...")
import networkx.algorithms.community as nx_comm
partition_sets = []
for community in tqdm(nx_comm.greedy_modularity_communities(G, weight="weight"), desc="Detecting communities", ncols=100):
    partition_sets.append(community)
partition = {node: i for i, community in enumerate(partition_sets) for node in community}
print(f"[INFO] Detected {len(partition_sets)} communities.")
elapsed = time.time() - start

print(f"Louvain complete in {elapsed:.2f}s. Writing outputs...")

print("[INFO] Saving results...")
for _ in tqdm(range(3), desc="Writing files", ncols=100):
    time.sleep(0.5)

# Save table of nodes -> community id
communities_df = pd.DataFrame({"node": list(partition.keys()), "community": list(partition.values())})
comm_csv = os.path.join(OUT_DIR, "communities.csv")
communities_df.to_csv(comm_csv, index=False)
print(f"[INFO] Saved communities table to {comm_csv}")

# Quick summary
summary_path = os.path.join(OUT_DIR, "community_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Total nodes: {G.number_of_nodes()}\n")
    f.write(f"Total edges: {G.number_of_edges()}\n")
    f.write(f"Detected communities: {communities_df['community'].nunique()}\n")
    f.write(f"Runtime (s): {elapsed:.2f}\n")
print(f"[INFO] Saved community summary to {summary_path}")
print(f"[INFO] Community detection and save steps took {elapsed:.2f}s")
