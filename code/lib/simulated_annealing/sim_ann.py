# Code for the simulated annealing algorithm.
# Coder: Eric Araujo
# Date of last changes: 2019-01-13

# To include the files from the other folders.
import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import lib.model.model as model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from multiprocessing.pool import Pool
from pprint import pprint


'''
Get the next neighbor to be used for the simulation.
'''
def neighbor(json_parameters):
    # inf_tau = -0.05
    # sup_tau = 0.05
    inf_tau = -0.01
    sup_tau = 0.01
    minn = 0.0001
    maxn_tau = 10

    maxn_mood = 1
    
    inf_sigma = -0.05
    sup_sigma = 0.05
    maxn_sigma = 200
    
    for key in json_parameters.keys():
        if key == 'mood_speed':
            mood_speed = json_parameters[key]
            inf_speed = -0.001
            sup_speed = 0.001
            new_mood_speed = mood_speed + ((sup_speed - inf_speed) * random() + inf_speed)
            new_mood_speed = minn if new_mood_speed < minn else maxn_mood if new_mood_speed > maxn_mood else new_mood_speed
            json_parameters[key] = new_mood_speed
        #elif key == 'like' or key == 'comment' or key == 'share':
        #    continue
        else:
            tau = json_parameters[key][0]
            new_tau = tau + ((sup_tau - inf_tau) * random() + inf_tau)
            new_tau = minn if new_tau < minn else maxn_tau if new_tau > maxn_tau else new_tau

            sigma = json_parameters[key][1]
            new_sigma = sigma + ((sup_sigma - inf_sigma) * random() + inf_sigma)
            new_sigma = minn if new_sigma < minn else maxn_sigma if new_sigma > maxn_sigma else new_sigma

            json_parameters[key] = [new_tau, new_sigma]

    return json_parameters


'''
Function to define acceptance probability values for SA
'''
def acceptance_probability(old_cost, new_cost, T):
    delta = (new_cost-old_cost)
    probability = np.exp(-delta/T)
    #print(-delta/T)
    # probability = 1/(1+np.exp(delta/T))
    return probability



'''
The empirical data should have a set of messages as input and different agents being simulated.

'''
def get_error(parameters=None, root_folder="../../"):
    mood = 0.5

    # Get the traits for the agents
    agents = model.get_agents(root_folder=root_folder) 
    
    # Agents
    try:
        agent1 = agents['1']
        agent2 = agents['2']
        agent3 = agents['3']
    except:
        print("Problems retrieving the traits of agent {} in the JSON.".format(agent))
        pprint(agents, indent=3)
        sys.exit(666)
    
    
    # Get validation data set
    validation_f = root_folder+'data/validation/'
    data_a1 = pd.read_csv(validation_f+'validation_agent_1.csv')
    data_a2 = pd.read_csv(validation_f+'validation_agent_2.csv')
    data_a3 = pd.read_csv(validation_f+'validation_agent_3.csv')
    
    messages1 = data_a1[[   'cat_per', 'cat_ent', 'cat_new', 'cat_edu', 
                            'cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 
                            'msg_sal', 'msg_med', 'msg_com', 'msg_que'
                        ]]
    messages2 = data_a2[[   'cat_per', 'cat_ent', 'cat_new', 'cat_edu', 
                            'cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 
                            'msg_sal', 'msg_med', 'msg_com', 'msg_que'
                        ]]
    messages3 = data_a3[[   'cat_per', 'cat_ent', 'cat_new', 'cat_edu', 
                            'cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 
                            'msg_sal', 'msg_med', 'msg_com', 'msg_que'
                        ]]
    
    reactions1 = data_a1[['mood', 'like', 'comment', 'share']]
    reactions2 = data_a2[['mood', 'like', 'comment', 'share']]
    reactions3 = data_a3[['mood', 'like', 'comment', 'share']]

    # Fitting the empirical data for a1 (reactions)
    reactions1 = reactions1.rename(columns=lambda x: x.strip())
    reactions1['time'] = (reactions1.index+1)*20-1
    reactions1.index = reactions1['time']
    reactions1 = reactions1[['mood', 'like', 'comment', 'share']]

    # Fitting the empirical data for a2 (reactions)
    reactions2 = reactions2.rename(columns=lambda x: x.strip())
    reactions2['time'] = (reactions2.index+1)*20-1
    reactions2.index = reactions2['time']
    reactions2 = reactions2[['mood', 'like', 'comment', 'share']]

    # Fitting the empirical data for a3 (reactions)
    reactions3 = reactions3.rename(columns=lambda x: x.strip())
    reactions3['time'] = (reactions3.index+1)*20-1
    reactions3.index = reactions3['time']
    reactions3 = reactions3[['mood', 'like', 'comment', 'share']]

    # Simulation
    # Agent 
    
    '''
    https://stackoverflow.com/questions/37873501/get-return-value-for-multi-processing-functions-in-python
    '''
    with Pool() as pool:
        result1 = pool.apply_async(model.run_message_sequence, (messages1, agent1,
                                                 parameters, 'nb1', root_folder))
        result2 = pool.apply_async(model.run_message_sequence, (messages2, agent2,
                                                 parameters, 'nb2', root_folder))
        result3 = pool.apply_async(model.run_message_sequence, (messages3, agent3,
                                                 parameters, 'nb3', root_folder))
        try:
            df1, df2, df3 = result1.get()[0], result2.get()[0], result3.get()[0]
            parameters = result1.get()[1]
        except:
            print(result1.get())
            exit(0)


    factor_diff = 3
    df1.index = df1.index.astype(int)
    dfpoints1 = df1[['mood', 'like', 'share', 'comment']].iloc[reactions1.index]
    #error1 = ((dfpoints1-reactions1)**2).sum().sum()
    error1 = (((dfpoints1[['like', 'share', 'comment']] - reactions1[['like', 'share', 'comment']])*factor_diff/factor_diff)**2).sum().sum() + ((dfpoints1['mood'] - (reactions1['mood'])*factor_diff)**2).sum().sum()

    df2.index = df2.index.astype(int)
    dfpoints2 = df2[['mood', 'like', 'share', 'comment']].iloc[reactions2.index]
    #error2 = ((dfpoints2-reactions2)**2).sum().sum()
    error2 = (((dfpoints2[['like', 'share', 'comment']] - reactions2[['like', 'share', 'comment']])*factor_diff/factor_diff)**2).sum().sum() + ((dfpoints2['mood'] - (reactions2['mood'])*factor_diff)**2).sum().sum()
    
    df3.index = df3.index.astype(int)
    dfpoints3 = df3[['mood', 'like', 'share', 'comment']].iloc[reactions3.index]
    #error3 = ((dfpoints3-reactions3)**2).sum().sum()
    error3 = (((dfpoints3[['like', 'share', 'comment']] - reactions3[['like', 'share', 'comment']])*factor_diff/factor_diff)**2).sum().sum() + ((dfpoints3['mood'] - (reactions3['mood'])*factor_diff)**2).sum().sum()

    #Calculate error
    sum_err = error1+error2+error3

    # the data frames are important for ploting later.
    return sum_err, parameters, dfpoints1, reactions1, dfpoints2, reactions2, dfpoints3, reactions3, df1, df2, df3




def plot_results(parameters, cost_hist, parameters_hist):

    sum_err, parameters, df1, empdf1, df2, empdf2, df3, empdf3 = get_error(parameters=parameters, plot=True)

    fig = plt.figure(figsize=((12, 6)))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Cost function', size=26, style="oblique", weight='bold')
    ax1.set_xlabel('Epochs', size=16)
    ax1.set_ylabel('Mean squared error (MSE)', size=16)
    ax1.tick_params(labelsize=14)
    ax1.plot(cost_hist)

    # ax2 = fig.add_subplot(3, 1, 2)
    # ax2.set_title('Parameters (edges weights)', size=26, style="oblique", weight='bold')
    # ax2.set_xlabel('Epochs', size=24)
    # ax2.set_ylabel('Value', size=24)
    # ax2.tick_params(labelsize=20)
    # ax2.legend(loc='best', title="")
    # df.plot(ax=ax2)

    # ax3 = fig.add_subplot(3, 1, 3)

    # # ax3.bar(index, dict(parameters).values())
    # df2 = pd.DataFrame(parameters)
    # df2.index = df2[0]
    # df2['positive'] = df2[1] > 0

    # # df2.plot(ax=ax3, kind='barh', color='green')
    # df2[1].plot(kind='barh', color=df2.positive.map({True: 'g', False: 'r'}))

    # ax3.set_title('Final parameters', size=26, fontname="Bold")
    # ax3.set_xlabel('Final value', size=24)
    # ax3.set_ylabel('Parameters', size=24)
    # ax3.tick_params(labelsize=20)

    fig.savefig('results.png')



def parameter_tuning(parameters=None, root_folder="../../"):
    # Keeping history (vectors)
    cost_hist = list([])
    parameters_hist = list([])

    # Actual cost
    old_cost, initial_parameters, _, _, _, _, _, _, _, _, _ = get_error(root_folder=root_folder)
    cost_hist.append(old_cost)
    parameters_hist.append(initial_parameters)

    T = 1.0
    T_min = 0.01
    # original = 0.9
    alpha = 0.7
    num_neighbors = 30
    parameters = initial_parameters

    while T > T_min:
        print('Temp: ', T)
        i = 1
        # original = 100
        while i <= num_neighbors:
            
            new_parameters = neighbor(parameters.copy())
            new_cost, new_parameters, _, _, _, _, _, _, _, _, _  = get_error(new_parameters.copy(), root_folder=root_folder)
            
            if new_cost < old_cost:
                print('Lower!')
                parameters = new_parameters.copy()
                parameters_hist.append(parameters.copy())
                old_cost = new_cost
                cost_hist.append(old_cost)
            else:
                ap = acceptance_probability(old_cost, new_cost, T)
                #print(ap)
                if ap > random():
                    #print('\n', new_parameters, '\n', new_cost, '\n')
                    print('accepted!')
                    parameters = new_parameters.copy()
                    parameters_hist.append(parameters.copy())
                    old_cost = new_cost
                    cost_hist.append(old_cost)
            i += 1
        pprint(parameters_hist[-1])
        print(cost_hist[-1])
        T = T*alpha

    # plot_results(parameters, cost_hist, parameters_hist)

    return parameters, cost_hist, parameters_hist

if __name__ == "__main__":
    parameter_tuning()

#fs_change, mood, -1