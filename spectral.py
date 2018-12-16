import networkx as nx
import scipy
from scipy.cluster import vq as vq
from scipy.sparse import csgraph as cs
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
from utils import objective_function
from sklearn.cluster import KMeans

folder = "data/test/"
path = "data/part1/ca-AstroPh.txt"


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
	if k == 2:
		eigM = v[:,1]
		
		labels = [0 if i<0 else 1 for i in eigM]
	else:
		objective = 1000000000
		eigM = v[:,:(k-1)]
		eigM = np.divide(eigM, np.reshape(np.linalg.norm(eigM, axis=1), (eigM.shape[0], 1)))
		for i in range(5):
			_, labels = vq.kmeans2(eigM,k)
			#kmeans = KMeans(n_clusters=k).fit(eigM)
			#labels = kmeans.labels_	
			obj = objective_function(G,labels)
			print(i, obj)
			if obj<objective:
				objective = obj
				best_labels = labels
		labels = best_labels
		
	return labels

'''
G = nx.karate_club_graph()
k = 2
labels = spectral_algorithm_normalized(G)
print(list(G.nodes()))
print(labels)
print("Objective: ",objective_function(G, labels))
print("Cluster sizes: ", Counter(labels))
print(list(G.edges()))
nx.draw(G, node_size=600, node_color=labels, with_labels=True, edge_color="red")
plt.savefig("Karate.pdf")
'''
files = os.listdir(folder)
for file in files:
	G = read_file_to_graph(folder+file)
	nodes = list(G.nodes())
	name, n_nodes, n_edges, k = read_header(folder+file)
	print(file)
	labels = spectral_algorithm_normalized(G)
	print("Cluster sizes: ", Counter(labels))
	file = open("clusters_"+file, "w")
	file.write("# "+name+" "+str(n_nodes)+" "+str(n_edges)+" "+str(k)+"\n")
	for i in range(n_nodes):
		file.write(str(nodes[i])+" "+str(labels[i])+"\n")
	file.close()
	print("Objective: ", objective_function(G, labels))
	#nx.draw(G, node_size=50, node_color=labels, edge_color="red")   #pos=nx.spring_layout(G, k=0.05, iterations=20)
	#plt.savefig(file[-4]+".pdf")



