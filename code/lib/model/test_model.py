import numpy as np
import networkx as nx
import pandas as pd
import math
#import matplotlib.pyplot as plt
import sys
from random import random
import model




g, w = model.generate_graph()

pprint(g.edges(data=True), indent=3)

msg_sequence = np.genfromtxt('data/messages.csv', delimiter=',', skip_header=1)

g1, w, traits, parameters, psd = model.run_message(message=msg_sequence[0], traits=np.random.rand(7), previous_status_dict=None,alogistic_parameters=None, speed_factor=0.5, delta_t=1, timesteps=30, weightList=None)

print(list(traits.values()))

g2, w, s, parameters, psd = model.run_message(message=msg_sequence[-1], traits=list(traits.values()), previous_status_dict=psd,alogistic_parameters=None, speed_factor=0.5, delta_t = 1, timesteps = 30, weightList=w)

inputsDF, parameters = model.run_message_sequence(message_seq=msg_sequence, traits=np.random.rand(7))

print(inputsDF.head())

print(parameters)

#print(w)

#print(s)

#print(parameters)

#print(psd)

print('\n***********************************\nTest simulation run successfully!')