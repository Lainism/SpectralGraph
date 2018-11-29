import networkx as nx
import scipy
from scipy.cluster import vq as vq
from scipy.sparse import csgraph as cs
import matplotlib.pyplot as plt
import numpy as np

path = "data/ca-GrQc.txt"

fh=open(path, 'rb')
G=nx.read_edgelist(fh)
fh.close()
num_nodes = len(G.nodes())
L = nx.normalized_laplacian_matrix(G)
print(L.shape)
w, v = scipy.sparse.linalg.eigsh(L, which="SM")
fiedler = v[:,1]
_, labels = vq.kmeans2(fiedler,2)
print(labels)
print(len(labels))

#figure = plt.figure()
#plt.plot(fiedler)
#plt.plot(np.sort(fiedler))
#plt.show()
#print(len(w))
#print(w)
#white = vq.whiten(v)
#means = vq.kmeans2(white, 2)
#print(len(G.nodes()))
#print(len(G.edges()))
#print (v[:,0])

nx.draw(G, node_size=50, node_color=labels, edge_color="red")   #pos=nx.spring_layout(G, k=0.05, iterations=20)
plt.savefig("graph.pdf")
#plt.show()
