import numpy as np
import networkx as nx
import pandas as pd

#import matplotlib.pyplot as plt

from random import random

edges_f = open('connections.csv')
nodes_f = open('states.csv')

graph = nx.DiGraph()


# Insert nodes
for line in nodes_f:
    node, func = line.replace(" ", "").strip().split(',')
    
    # Node not included
    if node not in graph.nodes():
        if node == 'es_a' or node == 'es_r':
            graph.add_node(node, attr_dict={'pos': 'output', 'func': func, 'status':{}} )
        elif func == 'id' or func == 'alogistic' or func == 'todefine':
            graph.add_node(node, attr_dict={'pos': 'inner', 'func': func, 'status':{}} )
        else:
            graph.add_node(node, attr_dict={'pos': 'input', 'func': func, 'status':{}} )
    else:
        print('<CONFLICT> Node already included in the list!')
        exit()

weightList=None
outWeightList = []

# Insert edges
if weightList is None:
    for line in edges_f:
        try:
            source, target, w = line.replace(" ", "").strip().split(',')
        except:
            print(line)
        # Comment this line if you don't wanna start with random values
        if w == 'oc':
            w = oc
        elif w == 'ja':
            w = ja
        else:
            w = float(w)*random()
        graph.add_edge(source, target, weight=float(w))
        outWeightList.append(((source, target), float(w)))
else:
    for line in weightList:
        ((source, target), w) = line
        graph.add_edge(source, target, weight=float(w))
        outWeightList.append(((source, target), float(w)))

'''
nx.draw_spring(graph, with_labels = True)
plt.draw()
#plt.show()
plt.savefig('graph_with_labels.png')
plt.clf()
'''