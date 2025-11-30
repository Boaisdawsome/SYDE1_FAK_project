import pickle, networkx as nx, matplotlib.pyplot as plt
G = pickle.load(open("outputs/syde1_community.gpickle","rb"))
plt.figure(figsize=(10,8))
nx.draw_networkx(G, node_size=10, with_labels=False)
plt.title("SYDE1–PTK2 Community")
plt.savefig("outputs/syde1_ptk2_cluster.png", dpi=300)