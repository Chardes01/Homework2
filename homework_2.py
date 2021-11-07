#!/usr/bin/env python
# coding: utf-8

# # 1. Preparation

# In[202]:


import numpy as np
from matplotlib import pyplot as plt


# In[2]:


def sigmoid (x, a = 1):
    return 1 / (1 + np.exp(-a * x))


# In[3]:


def sigmoidprime (x, a = 1):
    return sigmoid(x, a) * (1 - sigmoid(x, a))


# # 2 Data Set

# In[4]:


pinputs = np.array([[0,0],[0,1],[1,0],[1,1]])


# In[475]:


gates = {
'and' : np.array([0,0,0,1]),
'or'  : np.array([0,1,1,1]),
'nor' : np.array([1,0,0,0]),
'nand': np.array([1,1,1,0]),
'xor' : np.array([0,1,1,0])}


# # 3 Perceptron

# In[68]:


class Perceptron:
    
    def __init__(self, input_units):
        self.alpha = 1
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn()
    
    def forward_step(self, inputs):
        self.input = inputs
        return sigmoid(np.matmul(inputs, self.weights) + self.bias)
        
    def update(self, delta):
        self.bias = self.bias - self.alpha * delta
        self.weights = self.weights - self.alpha * delta * self.input
        return self.weights
    


# # 4 MLP

# In[449]:


class MLP:
    def __init__(self):
        # hidden layer of four perceptrons
        self.hidden_layer = np.array([
            Perceptron(2),
            Perceptron(2),
            Perceptron(2),
            Perceptron(2)
        ])
        # output layer of two perceptrons
        self.output_layer = Perceptron(self.hidden_layer.size)

    def forward_step(self, inputs):
        # initialize array of size of the hidden layer
        self.outputs = np.zeros(self.hidden_layer.size)
        # fill in the outputs of all hidden layer perceptrons
        for i, perceptron in enumerate(self.hidden_layer):
            self.outputs[i] = perceptron.forward_step(inputs)
        # pass on the output of the hidden layer to the output layer
            self.out = self.output_layer.forward_step(self.outputs)
        return self.out
        
    def backprop_step(self, label):
        # calculate the delta for the output layer
        delta_out = (self.out - label) * self.out * (1 - self.out)
        
        # calculate the deltas for the hidden layer and update the neurons
        for i, perceptron in enumerate(self.hidden_layer):
            delta_hidden = delta_out * self.output_layer.weights[i] * self.outputs[i] * (1 - self.outputs[i])
            perceptron.update(delta_hidden)
            
        # update the output neuron    
        self.output_layer.update(delta_out)
            
      


# # 5 Training & Visualization

# In[538]:


def training (gate):
    # get the label for the corresponding gate
    label = gates[gate]
    
    network = MLP()
    
    error = []
    accuracy = []
    epochs = []
    class_total = 0
    class_correct = 0
    
    # 1000 epochs
    for i in range(1000):
        epochs.append(i + 1)
        mse = 0
        
        # train MLP on each inputpossibility
        for n in range(4):
            classification = network.forward_step(pinputs[n])
            network.backprop_step(label[n])
            
            # calculate the resulting accuracy and error
            class_total += 1
            if (round(classification) == label[n]): class_correct += 1
            mse += pow(label[n] - classification,2)

        accuracy.append(class_correct/class_total)
        error.append(mse / 4)
    
    # plot the results
    plt.plot(epochs, error, accuracy)
    plt.legend(['Error', 'Accuracy'])
    plt.xlabel('epochs')
    plt.title(gate + '-gate-MLP')


# # 6 Visualization

# In[540]:
training('and')

