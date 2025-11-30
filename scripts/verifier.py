import pickle, networkx as nx
G = pickle.load(open("outputs/bipartite_network.gpickle","rb"))
print(G["SYDE1 (85360)"]["PTK2 (5747)"]["weight"])