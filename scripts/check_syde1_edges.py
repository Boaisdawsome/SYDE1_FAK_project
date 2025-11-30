import pickle, networkx as nx

G = pickle.load(open("outputs/bipartite_network.gpickle", "rb"))
syde1 = [n for n in G.nodes if "SYDE1" in str(n)]
print("SYDE1 degree:", G.degree(syde1[0]))
print("Neighbors:", list(G.neighbors(syde1[0]))[:20])