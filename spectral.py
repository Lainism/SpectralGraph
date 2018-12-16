import networkx as nx
import scipy
from scipy.cluster import vq as vq
from scipy.sparse import csgraph as cs

path = "data/ca-GrQc.txt"

fh=open(path, 'rb')

# Read the header
header = fh.readline()
headers = header.split(' ')
name = headers[1]
verticesn = int(headers[2])
edgesn = int(headers[3])
k = int(headers[4])

# Read the rest
G=nx.read_edgelist(fh)
fh.close()
L = nx.normalized_laplacian_matrix(G)
w, v = scipy.sparse.linalg.eigsh(L)
white = vq.whiten(v)
centroid, label = vq.kmeans2(white, k)
#print(len(G.nodes()))
#print(len(G.edges()))
print (v[:,0])
