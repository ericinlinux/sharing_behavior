import model as model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from multiprocessing.pool import Pool
from pprint import pprint


'''
'''
def neighbor(json_parameters):
    # inf_tau = -0.05
    # sup_tau = 0.05
    inf_tau = -0.05
    sup_tau = 0.05
    minn = 0.00001
    maxn_tau = 1
    
    inf_sigma = -0.1
    sup_sigma = 0.1
    maxn_sigma = 10
    
    for key in json_parameters.keys():
        if key == 'mood_speed':
            mood_speed = json_parameters[key]
            inf_speed = -0.1
            sup_speed = 0.1
            new_mood_speed = mood_speed + ((sup_speed - inf_speed) * random() + inf_speed)
            new_mood_speed = minn if new_mood_speed < minn else maxn_tau if new_mood_speed > maxn_tau else new_mood_speed
            json_parameters[key] = new_mood_speed
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
    delta = new_cost-old_cost
    probability = np.exp(-delta/T)
    return probability



'''
The empirical data should have a set of messages as input and different agents being simulated.

'''
def get_error(parameters=None):
    mood = 0.5

    # Get the traits for the agents
    validation_f = 'validation/'
    
    agent1 = pd.read_csv(validation_f+'agent_1.csv')
    agent2 = pd.read_csv(validation_f+'agent_2.csv')
    agent3 = pd.read_csv(validation_f+'agent_3.csv')

    a1_dict = agent1.to_dict()
    a2_dict = agent2.to_dict()
    a3_dict = agent3.to_dict()

    #[pt_con, nf_ko, nf_ent, nf_is, nf_si, nf_se, mood]
    a1_traits = [a1_dict['nf_ko'][0], a1_dict['nf_ent'][0], a1_dict['nf_is'][0], 
                 a1_dict['nf_si'][0], a1_dict['nf_se'][0], a1_dict['pt_con'][0], mood]
    a2_traits = [a2_dict['nf_ko'][0], a2_dict['nf_ent'][0], a2_dict['nf_is'][0], 
                 a2_dict['nf_si'][0], a2_dict['nf_se'][0], a2_dict['pt_con'][0], mood]
    a3_traits = [a3_dict['nf_ko'][0], a3_dict['nf_ent'][0], a3_dict['nf_is'][0], 
                 a3_dict['nf_si'][0], a3_dict['nf_se'][0], a3_dict['pt_con'][0], mood]
    
    
    # Get validation data set
    data_a1 = pd.read_csv(validation_f+'validation_agent_1.csv')
    data_a2 = pd.read_csv(validation_f+'validation_agent_2.csv')
    data_a3 = pd.read_csv(validation_f+'validation_agent_3.csv')
    
    messages1 = data_a1[['msg_cat_per', 'msg_cat_ent', 'msg_cat_new', 'msg_cat_edu', 
                        'msg_cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 
                        'msg_sal', 'msg_med', 'msg_com', 'msg_que']]
    messages2 = data_a2[['msg_cat_per', 'msg_cat_ent', 'msg_cat_new', 'msg_cat_edu', 
                        'msg_cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 
                        'msg_sal', 'msg_med', 'msg_com', 'msg_que']]
    messages3 = data_a3[['msg_cat_per', 'msg_cat_ent', 'msg_cat_new', 'msg_cat_edu', 
                        'msg_cat_con', 'msg_rel', 'msg_qua', 'msg_sen', 
                        'msg_sal', 'msg_med', 'msg_com', 'msg_que']]
    
    reactions1 = data_a1[['like', 'comment', 'share']]
    reactions2 = data_a2[['like', 'comment', 'share']]
    reactions3 = data_a3[['like', 'comment', 'share']]

    # Fitting the empirical data for a1 (reactions)
    reactions1 = reactions1.rename(columns=lambda x: x.strip())
    reactions1['time'] = (reactions1.index+1)*20-1
    reactions1.index = reactions1['time']
    reactions1 = reactions1[['like', 'comment', 'share']]

    # Fitting the empirical data for a2 (reactions)
    reactions2 = reactions2.rename(columns=lambda x: x.strip())
    reactions2['time'] = (reactions2.index+1)*20-1
    reactions2.index = reactions2['time']
    reactions2 = reactions2[['like', 'comment', 'share']]

    # Fitting the empirical data for a3 (reactions)
    reactions3 = reactions3.rename(columns=lambda x: x.strip())
    reactions3['time'] = (reactions3.index+1)*20-1
    reactions3.index = reactions3['time']
    reactions3 = reactions3[['like', 'comment', 'share']]

    # Simulation
    # Agent 
    '''
    df1, parameters = model.run_message_sequence(messages1.values, a1_traits,
                                                 alogistic_parameters=parameters, title='nb1')
    df2, parameters = model.run_message_sequence(messages2.values, a2_traits,
                                                 alogistic_parameters=parameters, title='nb2')
    df3, parameters = model.run_message_sequence(messages3.values, a3_traits,
                                                 alogistic_parameters=parameters, title='nb3')
    '''

    '''
    https://stackoverflow.com/questions/37873501/get-return-value-for-multi-processing-functions-in-python
    '''
    with Pool() as pool:
        result1 = pool.apply_async(model.run_message_sequence, (messages1.values, a1_traits,
                                                 parameters, 'nb1'))
        result2 = pool.apply_async(model.run_message_sequence, (messages2.values, a2_traits,
                                                 parameters, 'nb2'))
        result3 = pool.apply_async(model.run_message_sequence, (messages3.values, a3_traits,
                                                 parameters, 'nb3'))
        df1, df2, df3 = result1.get()[0], result2.get()[0], result3.get()[0]
        parameters = result1.get()[1]


    df1.index = df1.index.astype(int)
    dfpoints1 = df1[['like', 'share', 'comment']].iloc[reactions1.index]
    error1 = ((dfpoints1-reactions1)**2).sum().sum()

    df2.index = df2.index.astype(int)
    dfpoints2 = df2[['like', 'share', 'comment']].iloc[reactions2.index]
    error2 = ((dfpoints2-reactions2)**2).sum().sum()
    
    df3.index = df3.index.astype(int)
    dfpoints3 = df3[['like', 'share', 'comment']].iloc[reactions3.index]
    error3 = ((dfpoints3-reactions3)**2).sum().sum()

    #Calculate error
    sum_err = error1+error2+error3

    # the data frames are important for ploting later.
    return sum_err, parameters, dfpoints1, reactions1, dfpoints2, reactions2, dfpoints3, reactions3



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



def parameter_tuning(parameters=None):
    # Keeping history (vectors)
    cost_hist = list([])
    parameters_hist = list([])

    # Actual cost
    old_cost, initial_parameters, _, _, _, _, _, _ = get_error()
    cost_hist.append(old_cost)
    parameters_hist.append(initial_parameters)

    T = 1.0
    T_min = 0.01
    # original = 0.9
    alpha = 0.8
    parameters = initial_parameters

    while T > T_min:
        print('Temp: ', T)
        i = 1
        # original = 100
        while i <= 10:
            
            new_parameters = neighbor(parameters.copy())
            new_cost, new_parameters, _, _, _, _, _, _  = get_error(new_parameters)
            
            if new_cost < old_cost:
                parameters = new_parameters.copy()
                parameters_hist.append(parameters.copy())
                old_cost = new_cost
                cost_hist.append(old_cost)
            else:
                ap = acceptance_probability(old_cost, new_cost, T)
                if ap > random():
                    #print 'accepted!'
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