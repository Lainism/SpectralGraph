import networkx as nx
path = "ca-GrQc.txt"

fh=open(path, 'rb')
G=nx.read_edgelist(fh)
fh.close()
print(len(G.nodes()))
print(len(G.edges()))
