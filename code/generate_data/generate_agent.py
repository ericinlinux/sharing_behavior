import pandas as pd
import numpy as np
from random import randint


attributes = ['pt_con', 'nf_ko', 'nf_ent', 'nf_is', 'nf_si', 'nf_se',]
values = []


agent = {}
for i in range (len(attributes)):
	value = float(randint(1, 10))/10
	values.append(value)
	agent[attributes[i]] = values[i]


#Export output to .csv
filename = 'agent.csv'
output = pd.DataFrame()
output = output.append(agent, ignore_index=True)
output.to_csv(filename, index=False, index_label=False)