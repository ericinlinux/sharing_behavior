import pandas as pd
import numpy as np
import random
from random import randint



#Creating an empty dataframe for the messages
messages = pd.DataFrame()

features = ['cat_per', 'cat_ent', 'cat_new', 'cat_edu', 'cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 'msg_sal', 'msg_com','msg_med', 'msg_que']


#Creating 200 random messages

number_of_messages = 200

for i in range(number_of_messages):

	#Random category
	cat = randint(0, 4)
	cat_list = [0, 0, 0, 0, 0]
	cat_list[cat] = 1

	#Random values between 0 and 1
	values = list(np.random.uniform(low=0.05, high=1, size=5))

	#Media (none, picture, video)
	media = [0, 0.5, 1]
	msg_media = randint(0,2)
	values.append(media[msg_media])

	#Question (Randomly let 20% of the posts contain a question)
	msg_que = 0
	percentage = randint(0,10)
	if percentage > 8:
		msg_que = 1
	values.append(msg_que)

	
	#Create a list of al the vlaues
	values = cat_list + values

	#Create dictionary for features and values
	message = {}
	for i in range(len(features)):
		 message[features[i]] = values[i]

	#Append to dataframe
	messages = messages.append(message, ignore_index=True)


print(messages.head())


#Export dataframe to .csv
filename = 'messages.csv'
messages.to_csv(filename, index=False, index_label=False)




