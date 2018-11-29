import networkx as nx
import scipy
from scipy.cluster import vq as vq
from scipy.sparse import csgraph as cs

path = "ca-GrQc.txt"

fh=open(path, 'rb')
G=nx.read_edgelist(fh)
fh.close()
L = nx.normalized_laplacian_matrix(G)
w, v = scipy.sparse.linalg.eigsh(L)
white = vq.whiten(v)
means = vq.kmeans2(white, 2)
#print(len(G.nodes()))
#print(len(G.edges()))
print (v[:,0])
