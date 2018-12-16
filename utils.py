import networkx as nx
import numpy as np
from collections import Counter


def objective_function(graph, partition):
    down = min(Counter(partition).values())
    up = 0
    edges = list(graph.edges())
    nodes = list(graph.nodes())
    for edge in edges:
        start = edge[0]
        end = edge[1]
        start_label = partition[nodes.index(start)]
        end_label = partition[nodes.index(end)]
        if start_label != end_label:
            up += 1
    return float(up)/float(down)
