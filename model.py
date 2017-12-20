"""
Generate graph and run model for the political change model
Creator: Eric Araujo
Date: 2017-02-27
"""
'''
Parameters found using manual test on 14th March
parameters={
 "cons_content": [0.7, 2],
 "cons_eval": [0.7, 4],
 "cs_cons": [1.2, 2],
 "cs_lib": [1.2, 2],
 "fc_cons": [0.9, 2],
 "fc_lib": [0.9, 2],
 "lib_content": [0.7, 2],
 "lib_eval": [0.7, 4],
 "mood": [0.6, 2],
 "pp_cons": [1.4, 2],
 "pp_lib": [1.2, 2],
 "fs_h" : [0.9,5],
 "fs_s" : [0.9,2],
 "cs_speed": 0.00001,
 "mood_speed": 0.01,
 "fs_change" : [1.2, 4],
 "pp_speed": 0.001
}
'''


import sys
import numpy as np
import networkx as nx
import pandas as pd
import math
#import matplotlib.pyplot as plt
import json
from random import random


"""
Inputs: weightList with ((source,target),weight) values
        oc = openness/conscientiousness trait of the agent
        ja = justification system/adaptability trait of the agent
        traits is a vector with the information of [openness, adaptability, conscientiousness, system_justification] 
"""
def generate_graph(weightList=None, traits=None):
    # Files with edges and nodes
    try:
        edges_f = open('connections.csv')
        nodes_f = open('states.csv')
    except:
        print("Files for edges and nodes not included in the code folder!")
        exit(0)

    graph = nx.DiGraph()
    # Insert nodes
    for line in nodes_f:
        node, func = line.replace(" ", "").strip().split(',')
        # Node not included
        if node not in graph.nodes():
            if node == 'fs_change':
                graph.add_node(node, attr_dict={'pos': 'output', 'func': func, 'status':{}} )
            elif func in ['id', 'alogistic', 'alogistic+', 'diff', 'special']:
                graph.add_node(node, attr_dict={'pos': 'inner', 'func': func, 'status':{}} )
            else:
                graph.add_node(node, attr_dict={'pos': 'input', 'func': func, 'status':{}} )
        else:
            print('<CONFLICT> Node already included in the list!')
            exit()

    outWeightList = []
    
    # Insert edges
    if weightList == None:
        for line in edges_f:
            source, target, w = line.replace(" ", "").strip().split(',')
            # [openness, adaptability, conscientiousness, system_justification]
            if w == 'openness':
                w = traits[0]
            elif w == 'adaptability':
                w = traits[1]
            elif w == 'conscientiousness':
                w = traits[2]
            elif w == 'system_justification':
                w = traits[3]
            # In case w is negative, the value will follow
            else:
                w = float(w) #*random()
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))
    else:
        for line in weightList:
            ((source, target), w) = line

            '''pp_cons, cs_cons, conscientiousness
                cs_cons, pp_cons, system_justification
                pp_lib, cs_lib, openness
                cs_lib, pp_lib, adaptability
            '''
            if source == 'pp_cons' and target == 'cs_cons':
                w = traits[0]
            elif source == 'cs_cons' and target == 'pp_cons':
                w = traits[1]
            elif source == 'pp_lib' and target == 'cs_lib':
                w = traits[2]
            elif source == 'cs_lib' and target == 'pp_lib':
                w = traits[3]
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))

    #save_graph(graph)  

    return graph, outWeightList

def save_graph(graph):
    nx.draw_spring(graph, with_labels = True)
    plt.draw()
    #plt.show()
    plt.savefig('graph_with_labels.png')
    plt.clf()

"""
Input: value to calculate
tau: threshold
sigma: steepness
"""
def alogistic(c, tau, sigma):
    return ((1/(1+math.exp(-sigma*(c-tau))))-(1/(1+math.exp(sigma*tau))))*(1+math.exp(-sigma*tau))
    

"""
Inputs:     message sentiment, message political position, message quality - [msg_s, msg_p,msg_q]
            time of exposure - timesteps
            traits are [openness, adaptability, conscientiousness, system_justification] 
            alogistic_parameters is a dictionary with the tau and sigma for each node that uses alogistic 
            states should be a vector [pp_cons, pp_lib, cs_cons, cs_lib, mood] for the agent to start with
Outputs:    graph with the values for the states
            list of weights used to run the model
            return graph, outWeightList, set_output, alogistic_parameters
"""
def run_message(message=None, traits=None, states=None, previous_status_dict=None, alogistic_parameters=None, speed_factor=0.5, delta_t = 1, timesteps = 30, weightList=None):
    # Checking the values for the function
    if message is None or len(message) != 13:
        print('Pass the values of the message correctly to the function!')
        exit()
    if states is None or len(states) != 10:
        print('Pass the values of the states (pp, cs and mood) correctly to the function!')
        exit()
    #if previous_status_dict == None:
    #   print 'Starting from zero!'
        
    # Read the json file with the alogistic parameters
    if alogistic_parameters is None:
        try:
            with open('alogistic.json') as data_file:    
                alogistic_parameters = json.load(data_file)
        except:
            print('Couldn\'t read the alogistic parameters! Check the \'alogistic.json\' file!')
            exit()
    
    # Generate graph
    graph, outWeightList = generate_graph(weightList, traits)
    #print(graph.nodes(data=True))
    rng = np.arange(0.0, timesteps*delta_t, delta_t)
    pos = None
    for t in rng:
        # Initialize the nodes
        if t == 0:
            for node in graph.nodes():
                try:
                    func = graph.node[node]['attr_dict']['func']
                    pos = graph.node[node]['attr_dict']['pos']
                except:
                    print('node without func or pos %s at time %i' % (node, t))
                
                # Inputs receive a stable value for all the timesteps
                # message[0] is the time of the message
                if pos == 'input':
                    if node == 'msg_cat_per':
                        graph.node[node]['status'] = {0:message[1]}
                    elif node == 'msg_cat_ent':
                        graph.node[node]['status'] = {0:message[2]}
                    elif node == 'msg_cat_new':
                        graph.node[node]['status'] = {0:message[3]}
                    elif node == 'msg_cat_edu':
                        graph.node[node]['status'] = {0:message[4]}
                    elif node == 'msg_cat_con':
                        graph.node[node]['status'] = {0:message[5]}
                    elif node == 'msg_rel':
                        graph.node[node]['status'] = {0:message[6]}
                    elif node == 'msg_qua':
                        graph.node[node]['status'] = {0:message[7]}
                    elif node == 'msg_sen':
                        graph.node[node]['status'] = {0:message[8]}
                    elif node == 'msg_sal':
                        graph.node[node]['status'] = {0:message[9]}
                    elif node == 'msg_med':
                        graph.node[node]['status'] = {0:message[10]}
                    elif node == 'msg_com':
                        graph.node[node]['status'] = {0:message[11]}
                    elif node == 'msg_que':
                        graph.node[node]['status'] = {0:message[12]}
                    else:
                        print('Node with wrong value:', node)
                        exit()
                # states are the personality traits of the agent
                elif node == 'nf_ko':
                    graph.node[node]['status'] = {0:states[0]}
                elif node == 'nf_ent':
                    graph.node[node]['status'] = {0:states[1]}
                elif node == 'nf_is':
                    graph.node[node]['status'] = {0:states[2]}
                elif node == 'nf_si':
                    graph.node[node]['status'] = {0:states[3]}
                elif node == 'nf_si':
                    graph.node[node]['status'] = {0:states[4]}                
                elif node == 'nf_se':
                    graph.node[node]['status'] = {0:states[5]}
                elif node == 'pt_cons':
                    graph.node[node]['status'] = {0:states[6]}
                elif node == 'pt_agre':
                    graph.node[node]['status'] = {0:states[7]}
                elif node == 'pt_extra':
                    graph.node[node]['status'] = {0:states[8]}
                elif node == 'pt_neur':
                    graph.node[node]['status'] = {0:states[9]}
                # The other states are set to previous values at the beginning
                else:
                    if previous_status_dict is None:
                        graph.node[node]['status'] = {0:0}
                    else:
                        graph.node[node]['status'] = {0:previous_status_dict[node]}
            continue


        for node in graph.nodes():
            '''
                For each node (not 0 nodes...):
                    get the neighbors
                    get the function
                    get the weights for the edges
                    calculate the new status value for the node in time t
            '''
            func = graph.node[node]['func']
            pos = graph.node[node]['pos']

            # Get previous state
            previous_state = graph.node[node]['status'][t - delta_t]
            
            if pos != 'input':
                # If it is identity, the operation is based on the only neighbor.
                if func == 'id':
                    try:
                        weight = graph.edge[graph.predecessors(node)[0]][node]['weight']
                        state_pred = graph.node[graph.predecessors(node)[0]]['status'][t - delta_t]
                        if weight < 0:
                            graph.node[node]['status'][t] = previous_state + speed_factor * ((1-abs(weight) * state_pred) - previous_state) * delta_t
                        else:
                            graph.node[node]['status'][t] = previous_state + speed_factor * (weight * state_pred - previous_state) * delta_t
                    except:
                        print('<time ', t, '> node:', graph.predecessors(node)[0], '-> ', node, '(id)')
                        print(t - delta_t)
                    

                elif func == 'alogistic':
                    # This vector is the input for the alogistic function. It has the values to calculate it
                    values_v = []
                    for neig in graph.predecessors(node):
                        neig_w = graph.edge[neig][node]['weight']
                        neig_s = graph.node[neig]['status'][t - delta_t]
                        
                        values_v.append(neig_w*neig_s)
                    
                    tau = alogistic_parameters[node][0]
                    sigma = alogistic_parameters[node][1]
                    try:
                        c = max(0,alogistic(sum(values_v), tau, sigma))
                    except OverflowError as err:
                        print(err)
                    
                    # Changes for the speed factors
                    if node == 'pp_cons' or node == 'pp_lib':
                        sf = alogistic_parameters['pp_speed']
                    elif node == 'cs_cons' or node == 'cs_lib':
                        sf = alogistic_parameters['cs_speed']
                    elif node == 'mood':
                        sf = alogistic_parameters['mood_speed']
                    else:
                        sf = speed_factor

                    graph.node[node]['status'][t] = previous_state + sf * (c - previous_state) * delta_t


                    '''
                    cons_eval, alogistic+
                    lib_eval, alogistic+
                    pp_cons, alogistic+
                    pp_lib, alogistic+
                    '''
                elif func == 'alogistic+':
                    if node == 'cons_eval':
                        mood_w = graph.edge['mood'][node]['weight']
                        mood_s = graph.node['mood']['status'][t - delta_t]
                        mood_t = mood_w*mood_s

                        cons_acc_w = graph.edge['cons_acc'][node]['weight']
                        cons_acc_s = graph.node['cons_acc']['status'][t - delta_t]
                        cons_acc_t = cons_acc_w*cons_acc_s 

                        cons_content_w = graph.edge['cons_content'][node]['weight']
                        cons_content_s = graph.node['cons_content']['status'][t - delta_t]
                        cons_content_t = cons_content_w*cons_content_s 

                        tau = alogistic_parameters[node][0]
                        sigma = alogistic_parameters[node][1]
                        try:
                            c_values = [mood_t*(cons_content_t), (1-mood_t)*(cons_acc_t)]
                            c = alogistic(sum(c_values), tau, sigma)
                        except OverflowError as err:
                            print(err)
                        graph.node[node]['status'][t] = previous_state + speed_factor * (c - previous_state) * delta_t                      

                    elif node == 'lib_eval':
                        mood_w = graph.edge['mood'][node]['weight']
                        mood_s = graph.node['mood']['status'][t - delta_t]
                        mood_t = mood_w*mood_s

                        lib_acc_w = graph.edge['lib_acc'][node]['weight']
                        lib_acc_s = graph.node['lib_acc']['status'][t - delta_t]
                        lib_acc_t = lib_acc_w*lib_acc_s 

                        lib_content_w = graph.edge['lib_content'][node]['weight']
                        lib_content_s = graph.node['lib_content']['status'][t - delta_t]
                        lib_content_t = lib_content_w*lib_content_s 

                        tau = alogistic_parameters[node][0]
                        sigma = alogistic_parameters[node][1]
                        try:
                            c_values = [mood_t*(lib_content_t), (1-mood_t)*(lib_acc_t)]
                            c = alogistic(sum(c_values), tau, sigma)
                        except OverflowError as err:
                            print(err)
                        graph.node[node]['status'][t] = previous_state + speed_factor * (c - previous_state) * delta_t  
                    
                    else:
                        print('func todefine incorrect.\n', node, func, t)
                        exit()
                    '''
                    fs_cons, diff
                    fs_lib, diff
                    '''
                elif func == 'diff':
                    if node == 'fc_cons':
                        cons_eval_w = graph.edge['cons_eval'][node]['weight']
                        cons_eval_s = graph.node['cons_eval']['status'][t - delta_t]
                        cons_eval_t = cons_eval_w*cons_eval_s

                        pp_cons_w = graph.edge['pp_cons'][node]['weight']
                        pp_cons_s = graph.node['pp_cons']['status'][t - delta_t]
                        pp_cons_t = pp_cons_w*pp_cons_s

                        cs_cons_w = graph.edge['cs_cons'][node]['weight']
                        cs_cons_s = graph.node['cs_cons']['status'][t - delta_t]
                        cs_cons_t = cs_cons_w*cs_cons_s

                        # c_values = [abs(cons_eval_t - pp_cons_t), abs(cs_cons_t - pp_cons_t), abs(cs_cons_t - cons_eval_t)]
                        c_values = [abs(cons_eval_t - pp_cons_t), abs(cs_cons_t - cons_eval_t)]
                        # c_values = [abs(cons_eval_t - pp_cons_t), abs(cs_cons_t - pp_cons_t)]
                        tau = alogistic_parameters[node][0]
                        sigma = alogistic_parameters[node][1]

                        try:
                            c = alogistic(sum(c_values), tau, sigma)
                            #c = sum(c_values)/(cons_eval_w+pp_cons_w+cs_cons_w)
                        except OverflowError as err:
                            print(err)

                        graph.node[node]['status'][t] = previous_state + speed_factor * (c - previous_state) * delta_t  

                    elif node == 'fc_lib':
                        lib_eval_w = graph.edge['lib_eval'][node]['weight']
                        lib_eval_s = graph.node['lib_eval']['status'][t - delta_t]
                        lib_eval_t = lib_eval_w*lib_eval_s

                        pp_lib_w = graph.edge['pp_lib'][node]['weight']
                        pp_lib_s = graph.node['pp_lib']['status'][t - delta_t]
                        pp_lib_t = pp_lib_w*pp_lib_s

                        cs_lib_w = graph.edge['cs_lib'][node]['weight']
                        cs_lib_s = graph.node['cs_lib']['status'][t - delta_t]
                        cs_lib_t = cs_lib_w*cs_lib_s

                        # c_values = [abs(lib_eval_t - pp_lib_t), abs(cs_lib_t - pp_lib_t), abs(cs_lib_t - lib_eval_t)]
                        c_values = [abs(lib_eval_t - pp_lib_t), abs(cs_lib_t - lib_eval_t)]
                        # c_values = [abs(lib_eval_t - pp_lib_t), abs(cs_lib_t - pp_lib_t)]
                        tau = alogistic_parameters[node][0]
                        sigma = alogistic_parameters[node][1]

                        try:
                            c = alogistic(sum(c_values), tau, sigma)
                            # c = sum(c_values)/(lib_eval_w+pp_lib_w+cs_lib_w)
                        except OverflowError as err:
                            print(err)
                        graph.node[node]['status'][t] = previous_state + speed_factor * (c - previous_state) * delta_t  
                        
                    else:
                        print('Node ', node, ' out of its right placement.')
                        exit()
                '''
                elif func == 'special':
                    if node == 'fs_change':
                        fc_lib_w = graph.edge['fc_lib'][node]['weight']
                        fc_lib_s = graph.node['fc_lib']['status'][t - delta_t]
                        fc_lib_t = fc_lib_w*fc_lib_s

                        fc_cons_w = graph.edge['fc_cons'][node]['weight']
                        fc_cons_s = graph.node['fc_cons']['status'][t - delta_t]
                        fc_cons_t = fc_cons_w*fc_cons_s

                        c = (fc_lib_t + fc_cons_t )/2
                        graph.node[node]['status'][t] = previous_state + speed_factor * (c - previous_state) * delta_t  

                    else:
                        print 'so wrong!'
                # None of the others
                else:
                    print 'It shouldn\'t be here!'
                '''
            # In case of inputs, copy the previous state again
            else:
                graph.node[node]['status'][t] = graph.node[node]['status'][t - delta_t]

    psd = {}
    for node in graph.nodes():
        psd[node] = graph.node[node]['status'][t]

    set_output = {"pp_cons": graph.node['pp_cons']['status'][t],
                "pp_lib": graph.node['pp_lib']['status'][t],
                "cs_cons": graph.node['cs_cons']['status'][t],
                "cs_lib": graph.node['cs_lib']['status'][t],
                "mood": graph.node['mood']['status'][t],
                "fs_change": graph.node['fs_change']['status'][t],
                "fc_cons": graph.node['fc_cons']['status'][t],
                "fc_lib": graph.node['fc_lib']['status'][t],
                }
    return graph, outWeightList, set_output, alogistic_parameters, psd


'''
Run a sequence of messages for one agent with specific traits and an initial state
'''
def run_message_sequence(message_seq=None, traits=None, states=None, alogistic_parameters=None, title='0'):

    timesteps = 20
    delta_t = 1
    speed_factor = 0.8
    weightList=None
    
    # Initialize empty df
    inputsDF = pd.DataFrame()
    # previous_states_dict
    psd = None

    # Initial states of the agent (pp_cons, pp_lib, cs_cons, cs_lib, mood)
    # a1States = [0.1, 0.8, 0.2, 0.7, 0.5]

    for message in message_seq:
        if psd == None:
            g, w, s, parameters, psd = run_message(message=message, weightList=weightList, traits=traits, states=states, 
                alogistic_parameters=alogistic_parameters, speed_factor=speed_factor, delta_t = delta_t, timesteps = timesteps)
        else:
            states = [s['pp_cons'], s['pp_lib'], s['cs_cons'], s['cs_lib'], s['mood']]
            g, w, s, parameters, psd = run_message(message=message, weightList=weightList, traits=traits, states=states, previous_status_dict=psd,
                alogistic_parameters=alogistic_parameters, speed_factor=speed_factor, delta_t = delta_t, timesteps = timesteps)

        status_results = {}
        for node in g.nodes():
            status_results[node] = g.node[node]['status']

        inputsDF = inputsDF.append(pd.DataFrame(status_results), ignore_index=True)

    return inputsDF, parameters