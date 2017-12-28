import model as model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random



'''
Shape of the json file:
{   "fs_s": [0.5, 10],
    "fs_h": [0.5, 10],
    "mood": [0.5,2],
    "cons_acc": [0.5, 10],
    "lib_acc": [0.5, 10],
    "cons_content": [0.5,10],
    "lib_content": [0.5,10],
    "cons_eval": [0.5, 2],
    "lib_eval": [0.5, 2],
    "fc_lib": [0.5,10],
    "fc_cons": [0.5,10],
    "pp_cons": [0.5,5],
    "pp_lib": [0.5,5], 
    "cs_cons": [0.5, 5],
    "cs_lib": [0.5, 5],
    "fs_change": [0.5, 10],
    "es_a": [0.5,2],
    "es_r": [0.5,2]
}
'''
def neighbor(json_parameters):
    # inf_tau = -0.05
    # sup_tau = 0.05
    inf_tau = -0.05
    sup_tau = 0.05
    minn = 0.00001
    maxn_tau = 1
    
    inf_sigma = -0.5
    sup_sigma = 0.5
    maxn_sigma = 20
    
    for key in json_parameters.keys():
        if key == 'mood_speed':
            mood_speed = json_parameters[key]
            inf_speed = -0.01
            sup_speed = 0.01
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

            # json_parameters[key] = [new_tau, new_sigma]
            json_parameters[key] = [new_tau, new_sigma]
            # json_parameters[key] = [tau, new_sigma]

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

- Different agents can be simulated using different traits, which are 
    [openness, adaptability, conscientiousness, system_justification]
    pp_cons, cs_cons, conscientiousness
    cs_cons, pp_cons, system_justification
    pp_lib, cs_lib, openness
    cs_lib, pp_lib, adaptability

    So, for this, we can have three agents as scenarios:
        agent 1: [0.9, 0.9, 0.90.1, 0.1]
        agent 2: [0.1, 0.1, 0.9, 0.9]
        agent 3: [0.5, 0.5, 0.5, 0.5]

- We need to set a common start state for each agent which would suit them the best:
    [pp_cons, pp_lib, cs_cons, cs_lib, mood]
    
    agent 1: [0.1, 0.8, 0.2, 0.7, 0.5]
    agent 2: [0.75, 0.05, 0.8, 0.1, 0.5]
    agent 3: [0.1, 0.1, 0.15, 0.2, 0.5]

- Now we need a set of messages. Each message carries along with it:
    [msg_s, msg_p,msg_q]

    Sequence of 5x [happy, conservative, objective]: [1.0, 0.01 1.0, 0.01, 1.0, 0.01]
    Sequence of 5x [balanced, conservative, objective]: [0.5, 0.5, 1.0, 0.01, 1.0, 0.01]
    Sequence of 5x [happy, balanced, objective]: [1.0, 0.01, 0.5, 0.5, 1.0, 0.01]
    Sequence of 5x [happy, conservative, balanced]: [1.0, 0.01, 1.0, 0.01,  0.5, 0.5]

    Sequence of 5x [happy, liberal, objective]: [1.0, 0.01, 0.01, 1.0, 1.0, 0.01]
    Sequence of 5x [balanced, liberal, objective]: [0.5, 0.5, 0.01, 1.0, 1.0, 0.01]
    Sequence of 5x [happy, balanced, objective]: [1.0, 0.01, 0.5, 0.5, 1.0, 0.01]
    Sequence of 5x [happy, liberal, balanced]: [1.0, 0.01, 0.01, 1.0, 0.5, 0.5]
'''
def get_error(parameters=None, plot=False):
    #[openness, adaptability, conscientiousness, system_justification]
    a1Traits = [0.9, 0.9, 0.1, 0.1]
    a2Traits = [0.1, 0.1, 0.9, 0.9]
    a3Traits = [0.5, 0.5, 0.5, 0.5]

    # Initial states of the agent (pp_cons, pp_lib, cs_cons, cs_lib, mood)
    a1States = [0.1, 0.8, 0.2, 0.7, 0.5]
    a2States = [0.75, 0.05, 0.8, 0.1, 0.5]
    a3States = [0.1, 0.1, 0.15, 0.2, 0.5]

    data_f = "./data"
    msg_sequence = np.genfromtxt(data_f+'/messages.csv', delimiter=',', skip_header=1)
    #print msg_sequence
    sum_err = 0


    # Empirical Data
    empdf1 = pd.read_csv('./data/agent1.csv')
    empdf1 = empdf1.rename(columns=lambda x: x.strip())
    empdf1['time'] = (empdf1.index+1)*20-1
    empdf1.index = empdf1['time']
    empdf1 = empdf1[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib', 'mood', 'fs_change']]

    empdf2 = pd.read_csv('./data/agent2.csv')
    empdf2 = empdf2.rename(columns=lambda x: x.strip())
    empdf2['time'] = (empdf2.index+1)*20-1
    empdf2.index = empdf2['time']
    empdf2 = empdf2[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib', 'mood', 'fs_change']]

    empdf3 = pd.read_csv('./data/agent3.csv')
    empdf3 = empdf3.rename(columns=lambda x: x.strip())
    empdf3['time'] = (empdf3.index+1)*20-1
    empdf3.index = empdf3['time']
    empdf3 = empdf3[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib', 'mood', 'fs_change']]


    # Get results for each agent
    df1, parameters = model.run_message_sequence(msg_sequence, a1Traits, a1States, alogistic_parameters=parameters, title='1')
    df1.index = df1.index.astype(int)
    dfpoints1 = df1[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib', 'mood', 'fs_change']].iloc[empdf1.index]

    df2, parameters = model.run_message_sequence(msg_sequence, a2Traits, a2States, alogistic_parameters=parameters, title='2')
    df2.index = df2.index.astype(int)
    dfpoints2 = df2[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib', 'mood', 'fs_change']].iloc[empdf2.index]


    df3, parameters = model.run_message_sequence(msg_sequence, a3Traits, a3States, alogistic_parameters=parameters, title='3')
    df3.index = df3.index.astype(int)
    dfpoints3 = df3[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib', 'mood', 'fs_change']].iloc[empdf3.index]

    if plot:
        df1[['pp_cons', 'pp_lib', 'mood', 'cs_cons', 'cs_lib']].plot(figsize=((14,8)))
        plt.savefig('output_a1.png')
        plt.clf()
        # df1[['es_r', 'fs_change']].plot(figsize=((14,8)))
        df1[['fs_change', 'fc_cons', 'fc_lib']].plot(figsize=((14,8)))
        plt.savefig('output_a1e.png')
        plt.clf()

        df2[['pp_cons', 'pp_lib', 'mood', 'cs_cons', 'cs_lib']].plot(figsize=((14,6)))
        plt.savefig('output_a2.png')
        plt.clf()
        df2[['fs_change', 'fc_cons', 'fc_lib']].plot(figsize=((14,8)))
        plt.savefig('output_a2e.png')
        plt.clf()
        
        df3[['pp_cons', 'pp_lib', 'mood', 'cs_cons', 'cs_lib']].plot(figsize=((14,6)))
        plt.savefig('output_a3.png')
        plt.clf()
        df3[['fs_change']].plot(figsize=((14,8)))
        plt.savefig('output_a3e.png')
        plt.clf()

    #Calculate error
    sum_err = sum_err + \
        ((dfpoints1-empdf1)**2).sum().sum()+ ((dfpoints2-empdf2)**2).sum().sum() + ((dfpoints3-empdf3)**2).sum().sum()

        
    '''
    ((dfpoints1[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib']]-empdf1[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib']])**2).sum().sum()*1000 + \
        ((dfpoints1[['mood', 'fs_change']]-empdf1[['mood', 'fs_change']])**2).sum().sum() #+ \

    ((dfpoints2[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib']]-empdf2[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib']])**2).sum().sum()*100 + \
    ((dfpoints2[['mood', 'fs_change']]-empdf2[['mood', 'fs_change']])**2).sum().sum() + \
    ((dfpoints3[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib']]-empdf3[['pp_cons', 'pp_lib', 'cs_cons', 'cs_lib']])**2).sum().sum()*100 + \
    ((dfpoints3[['mood', 'fs_change']]-empdf3[['mood', 'fs_change']])**2).sum().sum() 
    '''

        # ((dfpoints2-empdf2)**2).sum().sum() + ((dfpoints3-empdf3)**2).sum().sum()
    # return sum_err, parameters, df1, empdf1, df2, empdf2, df3, empdf3
    return sum_err, parameters, df1, empdf1, df1, empdf1, df1, empdf1



def plot_results(parameters, cost_hist, parameters_hist):

    sum_err, parameters, df1, empdf1, df2, empdf2, df3, empdf3 = get_error(parameters=parameters, plot=True)
    '''
    i = 0
    new_dict = {}
    for item in parameters_hist:
        for parameter in item:
            if parameter[0] not in new_dict.keys():
                new_dict[parameter[0]] = []
            else:
                new_dict[parameter[0]].append(parameter[1])
        i += 1

    df = pd.DataFrame(new_dict)
    df.columns.names = ['source', 'target']
    '''

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
    old_cost, initial_parameters, df, empdf = get_error()
    cost_hist.append(old_cost)
    parameters_hist.append(initial_parameters)

    T = 1.0
    T_min = 0.01
    # original = 0.9
    alpha = 0.8
    parameters = initial_parameters

    while T > T_min:
        print 'Temp: ', T
        i = 1
        # original = 100
        while i <= 100:
            
            new_parameters = neighbor(parameters.copy())
            new_cost, new_parameters, df, empdf = get_error(new_parameters)
            
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
        print cost_hist[-1]
        T = T*alpha

    plot_results(parameters, cost_hist, parameters_hist)

    return parameters, cost_hist, parameters_hist

if __name__ == "__main__":
    parameter_tuning()

#fs_change, mood, -1