"""
Generate graph and run model for the sharing behavior on web media
Creator: Eric Araujo
Date: 2017-12-20
"""

import numpy as np
import networkx as nx
import pandas as pd
import math
import json


def generate_graph(weightList=None):
    """
    Inputs: weightList with ((source,target),weight) values
    """
    try:
        edges_f = open('connections.csv')
        nodes_f = open('states.csv')
    except:
        print("Files for edges and nodes not included in the code folder!")
        exit(0)
    # Initiate graph as digraph (oriented graph)
    graph = nx.DiGraph()
    # Insert nodes
    for line in nodes_f:
        # Read each line and split to get nodes' name and function
        node, func = line.replace(" ", "").strip().split(',')
        # Avoiding include repeated nodes
        if node not in graph.nodes():
            # If node is output
            if node in ['like', 'share', 'comment']:
                graph.add_node(node, attr_dict={'pos': 'output', 'func': func, 'status': {}})
            # If node is internal state
            elif func in ['id', 'alogistic']:
                graph.add_node(node, attr_dict={'pos': 'inner', 'func': func, 'status': {}})
            # If node is a trait of the participant
            elif func == 'trait':
                graph.add_node(node, attr_dict={'pos': 'trait', 'func': func, 'status': {}})
            # If node is an input
            elif func == 'input':
                graph.add_node(node, attr_dict={'pos': 'input', 'func': func, 'status': {}})
            else:
                print('Node %s does not match the requirements to create graph.', node)
                exit(0)
        else:
            print('<CONFLICT> Node %s already included in the list!', node)
            exit(0)

    outWeightList = []

    # Insert edges
    if weightList is None:
        for line in edges_f:
            source, target, w = line.replace(" ", "").strip().split(',')
            #w = float(w) #*random()
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))
    else:
        for line in weightList:
            ((source, target), w) = line
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))

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
            alogistic_parameters is a dictionary with the tau and sigma for each node that uses alogistic 
            states should be a vector [pp_cons, pp_lib, cs_cons, cs_lib, mood] for the agent to start with
Outputs:    graph with the values for the states
            list of weights used to run the model
            return graph, outWeightList, set_output, alogistic_parameters
"""
def run_message(message=None, traits=None, previous_status_dict=None,
                alogistic_parameters=None, speed_factor=0.5, delta_t=1,
                timesteps=30, weightList=None):
    # Checking the values for the function
    if message is None or len(message) != 13:
        print('Pass the values of the message correctly to the function!')
        exit()
    if traits is None or len(traits) != 10:
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
    graph, outWeightList = generate_graph(weightList)
    #print(graph.nodes(data=True))
    rng = np.arange(0.0, timesteps*delta_t, delta_t)
    pos = None
    for t in rng:
        # Initialize the nodes
        if t == 0:
            for node in graph.nodes():
                try:
                    func = graph.nodes[node]['attr_dict']['func']
                    pos = graph.nodes[node]['attr_dict']['pos']
                    #print(node, func, pos)
                except:
                    print('node without func or pos %s at time %i' % (node, t))
                
                # Inputs receive a stable value for all the timesteps
                # message[0] is the time of the message
                if pos == 'input':
                    if node == 'msg_cat_per':
                        graph.nodes[node]['status'] = {0:message[1]}
                    elif node == 'msg_cat_ent':
                        graph.nodes[node]['status'] = {0:message[2]}
                    elif node == 'msg_cat_new':
                        graph.nodes[node]['status'] = {0:message[3]}
                    elif node == 'msg_cat_edu':
                        graph.nodes[node]['status'] = {0:message[4]}
                    elif node == 'msg_cat_con':
                        graph.nodes[node]['status'] = {0:message[5]}
                    elif node == 'msg_rel':
                        graph.nodes[node]['status'] = {0:message[6]}
                    elif node == 'msg_qua':
                        graph.nodes[node]['status'] = {0:message[7]}
                    elif node == 'msg_sen':
                        graph.nodes[node]['status'] = {0:message[8]}
                    elif node == 'msg_sal':
                        graph.nodes[node]['status'] = {0:message[9]}
                    elif node == 'msg_med':
                        graph.nodes[node]['status'] = {0:message[10]}
                    elif node == 'msg_com':
                        graph.nodes[node]['status'] = {0:message[11]}
                    elif node == 'msg_que':
                        graph.nodes[node]['status'] = {0:message[12]}
                    else:
                        print('Node with wrong value:', node)
                        exit()
                # states are the personality traits of the agent
                elif node == 'nf_ko':
                    graph.nodes[node]['status'] = {0:traits[0]}
                elif node == 'nf_ent':
                    graph.nodes[node]['status'] = {0:traits[1]}
                elif node == 'nf_is':
                    graph.nodes[node]['status'] = {0:traits[2]}
                elif node == 'nf_si':
                    graph.nodes[node]['status'] = {0:traits[3]}
                elif node == 'nf_si':
                    graph.nodes[node]['status'] = {0:traits[4]}                
                elif node == 'nf_se':
                    graph.nodes[node]['status'] = {0:traits[5]}
                elif node == 'pt_cons':
                    graph.nodes[node]['status'] = {0:traits[6]}
                elif node == 'pt_agre':
                    graph.nodes[node]['status'] = {0:traits[7]}
                elif node == 'pt_extra':
                    graph.nodes[node]['status'] = {0:traits[8]}
                elif node == 'pt_neur':
                    graph.nodes[node]['status'] = {0:traits[9]}
                # The other states are set to previous values at the beginning
                else:
                    if previous_status_dict is None:
                        graph.nodes[node]['status'] = {0:0}
                    else:
                        graph.nodes[node]['status'] = {0:previous_status_dict[node]}
            continue

        for node in graph.nodes:
            '''
                For each node (not 0 nodes...):
                    get the neighbors
                    get the function
                    get the weights for the edges
                    calculate the new status value for the node in time t
            '''

            func = graph.nodes[node]['attr_dict']['func']
            pos = graph.nodes[node]['attr_dict']['pos']

            # Get previous state
            try:
                previous_state = graph.nodes[node]['status'][t - delta_t]
            except:
                print(graph.nodes[node]['status'], t, delta_t, node)
                print(graph.nodes[node]['attr_dict']['pos'])

            if pos != 'input' and pos != 'trait':
                # If it is identity, the operation is based on the only neighbor.
                if func == 'id':
                    try:
                        weight = graph.edges[list(graph.predecessors(node))[0], node]['weight']
                        state_pred = graph.nodes[list(graph.predecessors(node))[0]]['status'][t - delta_t]
                        if weight < 0:
                            graph.nodes[node]['status'][t] = previous_state + speed_factor * ((1-abs(weight) * state_pred) - previous_state) * delta_t
                        else:
                            graph.nodes[node]['status'][t] = previous_state + speed_factor * (weight * state_pred - previous_state) * delta_t
                    except:
                        #print('<time ', t, '> node:', list(graph.predecessors(node))[0], '-> ', node, '(id)')
                        print(node, list(graph.predecessors(node)))
                        print(t - delta_t)

                elif func == 'alogistic':
                    # This vector is the input for the alogistic function. It has the values to calculate it
                    values_v = []
                    for neig in graph.predecessors(node):
                        neig_w = graph.edges[neig, node]['weight']
                        neig_s = graph.nodes[neig]['status'][t - delta_t]

                        values_v.append(neig_w * neig_s)

                    tau = alogistic_parameters[node][0]
                    sigma = alogistic_parameters[node][1]
                    try:
                        c = max(0, alogistic(sum(values_v), tau, sigma))
                    except OverflowError as err:
                        print(err)

                    # Changes for the speed factors
                    if node == 'mood':
                        sf = alogistic_parameters['mood_speed']
                    else:
                        sf = speed_factor

                    graph.nodes[node]['status'][t] = previous_state + sf * (c - previous_state) * delta_t
                
            # In case of inputs, copy the previous state again
            else:
                graph.nodes[node]['status'][t] = graph.nodes[node]['status'][t - delta_t]

    # Previous status dictionary to keep track of what was done
    psd = {}
    for node in graph.nodes():
        psd[node] = graph.nodes[node]['status'][t]

    #
    set_output = {"nf_ko": graph.nodes['nf_ko']['status'][t],
                  "nf_ent": graph.nodes['nf_ent']['status'][t],
                  "nf_is": graph.nodes['nf_is']['status'][t],
                  "nf_si": graph.nodes['nf_si']['status'][t],
                  "nf_se": graph.nodes['nf_se']['status'][t],
                  "pt_cons": graph.nodes['pt_cons']['status'][t],
                 }
    return graph, outWeightList, set_output, alogistic_parameters, psd


def run_message_sequence(message_seq=None, traits=None, states=None, alogistic_parameters=None, title='0'):
    '''
    Run a sequence of messages for one agent with specific traits and an initial state
    '''
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
        if psd is None:
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