import pandas as pd
import numpy as np
import random
import csv
from random import randint



#Import agant
agent = {}
with open('agent_3.csv') as csvfile:
	agent_csv = csv.reader(csvfile, delimiter=',')
	agent_csv = list(agent_csv)
	for i in range(len(agent_csv[0])):
		 agent[agent_csv[0][i]] = agent_csv[1][i]


#Creating an empty dataframe for the messages
messages = pd.DataFrame()

features = ['cat_per', 'cat_ent', 'cat_new', 'cat_edu', 'cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 'msg_sal', 'msg_com','msg_med', 'msg_que']
outputs =  ['like', 'comment', 'share']


#Creating 200 random messages

number_of_messages = 100

for i in range(number_of_messages):

	#Random category
	cat = randint(0, 4)
	cat_list = [0, 0, 0, 0, 0]
	cat_list[cat] = 1

	#five random values between 0 and 1
	values = list(np.random.uniform(low=0.05, high=1, size=5))

	#Media (none, picture, video)
	media = [0, 0.5, 1]
	msg_media = randint(0,2)
	values.append(media[msg_media])

	#Question (Randomly let 20% of the posts contain a question)
	msg_que = 0
	percentage = randint(1,10)
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




#Generating outputs

likes = []
comments = []
shares = []


pt_con = float(agent['pt_con'])

for i in range(number_of_messages):
	points = float(messages.loc[i]['msg_rel']) + float(messages.loc[i]['msg_qua']) + float(messages.loc[i]['msg_sal']) + float(messages.loc[i]['msg_med']) + float(messages.loc[i]['msg_sen'])        
	if (float((agent['nf_ko'])) + (messages.loc[i]['cat_per'])) >= 1.5:
		points += 1
	if (float((agent['nf_ent'])) + (messages.loc[i]['cat_ent'])) >= 1.5:
		points += 1
	if (float((agent['nf_is'])) + (messages.loc[i]['cat_new'])) >= 1.5:
		points += 1	
	if (float((agent['nf_is'])) + (messages.loc[i]['cat_edu'])) >= 1.5:
		points += 1
	if (float((agent['nf_si'])) + (messages.loc[i]['cat_con'])) >= 1.5:
		points += 1
	
	if points  >= 3.5:
		percentage = randint(1,10)
		if percentage > (pt_con*10):
			likes.append(1)
			percentage = randint(1,10)
			if percentage > 7:
				shares.append(1)
			else:
				shares.append(0)
		else:
			likes.append(0)
			shares.append(0)


		comment_points = float(messages.loc[i]['msg_com']) + float(agent['nf_si']) + float(messages.loc[i]['msg_rel'])
		if comment_points >= 2:
			percentage = randint(1,50)
			if messages.loc[i]['msg_que'] == 1.0:
				percentage = (percentage * 1.8)
			if percentage > (30 + (pt_con*10)):
				comments.append(1)
			else:
				comments.append(0)
		else:
			comments.append(0)


	else:
		likes.append(0)
		comments.append(0)
		shares.append(0)


messages['like'] = likes
messages['comment'] = comments
messages['share'] = shares




#Export dataframe to .csv
filename = 'validation_agent_3.csv'
messages.to_csv(filename, index=False, index_label=False)
