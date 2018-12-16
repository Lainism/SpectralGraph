import networkx as nx
import scipy
from scipy.cluster import vq as vq
from scipy.sparse import csgraph as cs
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

path = "data/part2/Oregon-1.txt"


def read_header(path):
	fh = open(path, 'rb')
	header = fh.readline()
	headers = header.split(' ')
	name = headers[1]
	verticesn = int(headers[2])
	edgesn = int(headers[3])
	if len(headers) == 4:
		k = 2
	else:
		k = int(headers[4])
	fh.close()
	return name, verticesn, edgesn, k

def read_file_to_graph(path):
	fh = open(path, 'rb')
	G = nx.read_edgelist(fh)
	fh.close()
	return G

def spectral_algorithm(graph):
	L = nx.laplacian_matrix(G)
	L = L.astype("float32")
	w, v = scipy.sparse.linalg.eigsh(L, which="SM")
	eigM = v[:,:(k-1)]
	_, labels = vq.kmeans2(eigM,k)
	return labels

def spectral_algorithm_normalized(graph):
	L = nx.normalized_laplacian_matrix(G)
	w, v = scipy.sparse.linalg.eigsh(L, which="SM")
	eigM = v[:,:(k-1)]
	norm_eigM = np.divide(eigM, np.reshape(np.linalg.norm(eigM, axis=1), (eigM.shape[0], 1)))
	_, labels = vq.kmeans2(norm_eigM,k)
	return labels

G = read_file_to_graph(path)
name, n_nodes, n_edges, k = read_header(path)
labels = spectral_algorithm_normalized(G)
print(Counter(labels))

nx.draw(G, node_size=50, node_color=labels, edge_color="red")   #pos=nx.spring_layout(G, k=0.05, iterations=20)
plt.savefig("graph.pdf")

