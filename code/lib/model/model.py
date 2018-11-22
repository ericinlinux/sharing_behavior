"""
Generate graph and run model for the sharing behavior on web media
Creator: Eric Araujo
Date of Creation: 2018-10-19
Last update: 2018-11-20
"""

import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pprint import pprint

# To include the files from the other folders.
import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import lib.messages.generate_messages as gm

#from random import random


def get_agents(root_folder = "../../"):
    # Load agents
    agents_file = root_folder + "data/agents/agents.json"
    with open(agents_file, 'r') as f:
        agents = json.load(f)
    return agents


def generate_graph(weightList=None, root_folder='../../'):
    """
    Inputs: weightList with ((source,target),weight) values
    """
    try:
        edges_f = open(root_folder + 'data/model/model_connections.csv')
        nodes_f = open(root_folder+ 'data/model/model_states.csv')
    except:
        print("<FILES NOT FOUND>: model_connections.csv and model_states.csv not included in the data folder!")
        sys.exit(0)

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
                graph.add_node(node)
                graph.nodes()[node]['pos']='output'
                graph.nodes()[node]['func']=func
                graph.nodes()[node]['status'] = {}

            # If node is internal state
            elif func in ['id', 'alogistic']:
                graph.add_node(node)
                graph.nodes()[node]['pos'] = 'inner'
                graph.nodes()[node]['func'] = func
                graph.nodes()[node]['status'] = {}

            # If node is a trait of the participant
            elif func == 'trait':
                graph.add_node(node)
                graph.nodes()[node]['pos'] = 'trait'
                graph.nodes()[node]['func'] = func
                graph.nodes()[node]['status'] = {}

            # If node is an input
            elif func == 'input':
                graph.add_node(node)
                graph.nodes()[node]['pos'] = 'input'
                graph.nodes()[node]['func'] = func
                graph.nodes()[node]['status'] = {}

            else:
                print('Node %s does not match the requirements to create graph.', node)
                sys.exit(0)
        else:
            print('<CONFLICT> Node %s already included in the list!', node)
            sys.exit(0)

    outWeightList = []

    # Insert edges
    if weightList is None:
        for line in edges_f:
            source, target, w = line.replace(" ", "").strip().split(',')

            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))
    # In case you have changes in the edges over time.
    else:
        for line in weightList:
            ((source, target), w) = line
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))

    #print("Graph generated successfully. It contains {} nodes and {} edges.".format(
    #    graph.number_of_nodes(),graph.number_of_edges()))
    return graph, outWeightList


def save_graph(graph):
    """
    Networkx draw function does not look nice.
    Function needs improvements in the future.
    """
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
def run_message(message=None, traits=None,
                previous_status_dict=None, alogistic_parameters=None, 
                speed_factor=0.5, delta_t=1, timesteps=20, 
                weightList=None, root_folder="../../"):

    # Checking the values for the function
    if message is None or len(message) != 12:
        print('Pass the values of the message correctly to the function!')
        print(message)
        print(message.shape)
        sys.exit()
        
    # Read the json file with the alogistic parameters
    if alogistic_parameters is None:
        try:
            with open(root_folder+'data/model/alogistic_parameters.json') as data_file:    
                alogistic_parameters = json.load(data_file)
        except:
            print('Couldn\'t read the alogistic parameters! Check the \'alogistic.json\' file!')
            sys.exit()
    elif alogistic_parameters == 'random':
        alogistic_parameters = {
                     "srs_sal": [random()*10, random()*20],
                     "arousal": [0.45, random()*20],
                     "attention_1": [2.23, random()*20],
                     "attention_2": [0.23, random()*20],
                     "mood": [5.3, random()*20],
                     "ff_ko": [1.75, random()*20],
                     "ff_ent": [1.43, random()*20],
                     "ff_si": [1.12, random()*20],
                     "ff_is": [2.04, random()*20],
                     "ff_se": [2.45, random()*20],
                     "satisfaction": [2.1, random()*20],
                     "prep_like" : [2.8,random()*20],
                     "prep_comm": [3.5,random()*20],
                     "prep_share" : [2.1, random()*20],
                     "mood_speed": random()
                    }
    graph, outWeightList = generate_graph(weightList, root_folder=root_folder)

    rng = np.arange(0.0, timesteps*delta_t, delta_t)
    pos = None

    for t in rng:
        # Initialize the nodes on time 0
        if t == 0:
            for node in graph.nodes():
                try:
                    func = graph.nodes[node]['func']
                    pos = graph.nodes[node]['pos']
                    #print(node, func, pos)
                except:
                    print('node without func or pos %s at time %i' % (node, t))

                # Inputs receive a stable value for all the timesteps
                # message[0] is the time of the message
                if pos == 'input':
                    if node == 'cat_per':
                        graph.nodes[node]['status'] = {0:message['cat_per']}
                    elif node == 'cat_ent':
                        graph.nodes[node]['status'] = {0:message['cat_ent']}
                    elif node == 'cat_new':
                        graph.nodes[node]['status'] = {0:message['cat_new']}
                    elif node == 'cat_edu':
                        graph.nodes[node]['status'] = {0:message['cat_edu']}
                    elif node == 'cat_con':
                        graph.nodes[node]['status'] = {0:message['cat_con']}
                    elif node == 'msg_rel':
                        graph.nodes[node]['status'] = {0:message['msg_rel']}
                    elif node == 'msg_qua':
                        graph.nodes[node]['status'] = {0:message['msg_qua']}
                    elif node == 'msg_sen':
                        graph.nodes[node]['status'] = {0:message['msg_sen']}
                    elif node == 'msg_sal':
                        graph.nodes[node]['status'] = {0:message['msg_sal']}
                    elif node == 'msg_med':
                        graph.nodes[node]['status'] = {0:message['msg_med']}
                    elif node == 'msg_com':
                        graph.nodes[node]['status'] = {0:message['msg_com']}
                    elif node == 'msg_que':
                        graph.nodes[node]['status'] = {0:message['msg_que']}
                    else:
                        print('Node with wrong value:', node)
                        sys.exit()
                # states are the personality traits of the agent
                elif node == 'nf_ko':
                    graph.nodes[node]['status'] = {0:traits['nf_ko']}
                elif node == 'nf_ent':
                    graph.nodes[node]['status'] = {0:traits['nf_ent']}
                elif node == 'nf_is':
                    graph.nodes[node]['status'] = {0:traits['nf_is']}
                elif node == 'nf_si':
                    graph.nodes[node]['status'] = {0:traits['nf_si']}
                elif node == 'nf_se':
                    graph.nodes[node]['status'] = {0:traits['nf_se']}      
                elif node == 'pt_cons':
                    graph.nodes[node]['status'] = {0:traits['pt_cons']}
                elif node == 'mood':
                    graph.nodes[node]['status'] = {0:traits['mood']}
                # The other states are set to previous values at the beginning
                else:
                    if previous_status_dict is None:
                        graph.nodes[node]['status'] = {0:0}
                    # Keeping the state of the nodes from previous timestep
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

            func = graph.nodes[node]['func']
            pos = graph.nodes[node]['pos']

            # Get previous state
            try:
                previous_state = graph.nodes[node]['status'][t - delta_t]
            except:
                print(graph.nodes[node]['status'], t, delta_t, node)
                print(graph.nodes[node]['pos'])

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
    previous_states_dict = {}

    for node in graph.nodes():
        previous_states_dict[node] = graph.nodes[node]['status'][t]

    # all these states (apart from mood) should be the same over the simulation
    set_traits = {"nf_ko": graph.nodes['nf_ko']['status'][t],
                  "nf_ent": graph.nodes['nf_ent']['status'][t],
                  "nf_is": graph.nodes['nf_is']['status'][t],
                  "nf_si": graph.nodes['nf_si']['status'][t],
                  "nf_se": graph.nodes['nf_se']['status'][t],
                  "pt_cons": graph.nodes['pt_cons']['status'][t],
                  "mood": graph.nodes['mood']['status'][t],
                 }
    return graph, outWeightList, set_traits, alogistic_parameters, previous_states_dict


def run_message_sequence(message_seq=None, traits=None, alogistic_parameters=None, title='0', root_folder="../../"):
    '''
    Run a sequence of messages for one agent with specific traits and an initial state
    message_seq: array of messages
    traits:
    alogistic_parameters:
    title: Title for graphics to be plotted.
    '''
    timesteps = 20
    delta_t = 1 
    speed_factor = 0.8
    weightList=None
    
    # Initialize empty df
    inputsDF = pd.DataFrame()
    # previous_states_dict
    psd = None

    # message_seq is a DF. Convert it to DICT and get the items
    for idx, message in dict(message_seq.T).items():
        #print("Processing message: ", message)

        if psd is None:
            g, w, set_traits, parameters, psd = run_message(message=message, weightList=weightList, 
                traits=traits, alogistic_parameters=alogistic_parameters, speed_factor=speed_factor, delta_t=delta_t, timesteps=timesteps, root_folder=root_folder
                )
        else:
            g, w, set_traits, parameters, psd = run_message(message=message, weightList=weightList,
                    traits=set_traits, previous_status_dict=psd, 
                    alogistic_parameters=alogistic_parameters, speed_factor=speed_factor, 
                    delta_t=delta_t, timesteps=timesteps, root_folder=root_folder
                    )
        #print("Set of traits: ", set_traits)
        status_results = {}
        for node in g.nodes():
            status_results[node] = g.node[node]['status']

        inputsDF = inputsDF.append(pd.DataFrame(status_results), ignore_index=True)

    return inputsDF, parameters

if __name__ == "__main__":
    # Get messages
    print("Running tests for model.py code.")
    messages = gm.sequence_messages()
    message = dict(messages.iloc[0])
    last_message = dict(messages.iloc[-1])
    # Get graph
    g, w = generate_graph()
    
    # Get agents traits
    agent='1'
    agents = get_agents() 
    # Agents
    try:
        agent_traits = agents[agent]
    except:
        print("Problems retrieving the traits of agent {} in the JSON.".format(agent))
        pprint(agents, indent=3)
        sys.exit(666)

    g1, w, traits, parameters, psd = run_message(
                                                message=message, 
                                                traits=agent_traits, 
                                                previous_status_dict=None, 
                                                alogistic_parameters=None, 
                                                speed_factor=0.5, 
                                                delta_t=1, timesteps=20, 
                                                weightList=None
                                            )
    print(g1.nodes['mood'])

    g2, w, s, parameters, psd = run_message(message=last_message, 
                                              traits=traits, 
                                              previous_status_dict=psd,
                                              alogistic_parameters=parameters, 
                                              speed_factor=0.5, 
                                              delta_t = 1, timesteps = 30, 
                                              weightList=w
                                             )
    inputsDF, parameters = run_message_sequence(message_seq=messages, traits=traits,  
                                                        alogistic_parameters=None)
    print("End of Test.")