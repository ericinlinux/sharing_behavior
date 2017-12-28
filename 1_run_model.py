import numpy as np
import networkx as nx
import pandas as pd
import math
#import matplotlib.pyplot as plt
import sys
from random import random
import model
from pprint import pprint



g, w = model.generate_graph()

#pprint(g.edges(data=True), indent=2)
msg_sequence = np.genfromtxt('data/messages.csv', delimiter=',', skip_header=1)

# [1, 0.2, 0.8, 0.5, 0.2, 0.6, 1, 0.01, 0]

#g1, w, traits, parameters, psd = model.run_message(message=[1, 0, 1, 0, 0, 0, 0.8, 0.5, 0.2, 0.6, 1, 0.01, 0], traits=np.random.rand(7), previous_status_dict=None,alogistic_parameters=None, speed_factor=0.5, delta_t=1, timesteps=30, weightList=None)

# print(list(traits.values()))

#g2, w, s, parameters, psd = model.run_message(message=[1, 0, 1, 0, 0, 0, 0.8, 0.5, 0.2, 0.6, 1, 0.01, 0], traits=list(traits.values()), previous_status_dict=psd,alogistic_parameters=None, speed_factor=0.5, delta_t = 1, timesteps = 30, weightList=w)

inputsDF, parameters = model.run_message_sequence(message_seq=msg_sequence, traits=np.random.rand(7))

print(inputsDF.head())

print(parameters)

#run_message_sequence(message_seq=None, traits=None, alogistic_parameters=None, title='0'):
#print(w)

#print(s)

#print(parameters)

#print(psd)