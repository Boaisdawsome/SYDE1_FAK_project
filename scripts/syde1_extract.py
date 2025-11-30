import os
import pickle
import pandas as pd
import networkx as nx

OUT_DIR = "outputs"
net_path = os.path.join(OUT_DIR, "bipartite_network.gpickle")
comm_path = os.path.join(OUT_DIR, "community_assignments.csv")

# Load community map and graph
print("[INFO] Loading network and community assignments...")
with open(net_path, "rb") as f:
    G = pickle.load(f)
communities = pd.read_csv(comm_path)

# Locate SYDE1 node
syde1_nodes = [n for n in G.nodes if "SYDE1" in str(n).upper()]
if not syde1_nodes:
    raise ValueError("SYDE1 not found in network nodes.")
syde1_node = syde1_nodes[0]
print(f"[INFO] Found SYDE1 node: {syde1_node}")

# Find SYDE1's community ID
syde1_comm = communities.loc[communities["node"] == syde1_node, "community"].values[0]
print(f"[INFO] SYDE1 belongs to community {syde1_comm}")

# Extract all nodes in the same community
syde1_group = communities[communities["community"] == syde1_comm]["node"].tolist()
subG = G.subgraph(syde1_group).copy()
print(f"[INFO] Extracted subgraph: {len(subG.nodes())} nodes, {len(subG.edges())} edges")

# Save subgraph
subgraph_path = os.path.join(OUT_DIR, "syde1_community.gpickle")
with open(subgraph_path, "wb") as f:
    pickle.dump(subG, f)

# Save node list
pd.DataFrame({"node": syde1_group}).to_csv(os.path.join(OUT_DIR, "syde1_community_nodes.csv"), index=False)
print(f"[SUCCESS] Saved SYDE1 community subgraph and node list to outputs/")