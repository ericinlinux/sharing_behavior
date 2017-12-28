# Cognitive model for sharing behavior

This code is related to the article 'Sharing behavior in web media: a cognitive model proposal'.

The code is a simulation of the model provided in the article. 

![model](final_model.png)

## Structure of the code

The project contains two parts: the model and the parameter tuning code.

### The model

The file model.py contains all the functions to run the model. The code basically contains as inputs the messages received by the agent, its traits and the list of nodes and edges for the model.

As output the information about the reactions are provided by the states share, comment or like. The model is based on Facebook platform interaction by the user.

A function to run a sequence of messages is also provided. In this case, the values for speed factor, steps and step size are inside the function. The traits permit that different agents are simulated within different function calls.

The folder **data** contains the inputs for the model. The file *states.csv* contains the nodes present in the model and the function they use to calculate the next time steps values. The function *id* is based on the previous state connected to the node. The function *alogistic* depends on the values for threshold and steepness. The function can be seen in the article, and is calculated by the function **alogistic**(c, tau, sigma). The values for the parameters are in the *alogistic.json* file. The states that have input as a function are the nodes who are going to receive the values through the function. So they do not need to have a function, as the input is given. When the function is trait, then it means that the state is related to a trait of the agent. The traits are stable values that do not change over time.

The file *connections.csv* contains the connections between the nodes and the weight of them. Nodes with negative weights cause reverse effect in the following node.

### The parameter tuning

This part of the code is found in the file sim_ann.py. It contains the functions to run a simulated annealing algorithm to find the best parameters for the code. In our model we are tuning the parameters for the logistic function and the speed factor for the mood.



## Running the code

To run the model, with the basic test, run the command in the terminal:

```
python test_model.py
```
