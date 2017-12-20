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


# [1, 0.2, 0.8, 0.5, 0.2, 0.6, 1, 0.01, 0]

model.run_message(message=[1, 0.2, 0.8, 0.5, 0.2, 0.6, 1, 0.01, 0], 
                  traits=None, states=np.random.rand(10), previous_status_dict=None, 
                  alogistic_parameters=None, 
                  speed_factor=0.5, delta_t = 1, timesteps = 30, weightList=None)