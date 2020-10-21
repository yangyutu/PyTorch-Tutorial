# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:50:21 2019

@author: yuguangyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1)


LR = 0.02
DOWNLOAD_MNIST = False

# input of RNN layer are: input. h_0
#input of shape ( batch, seq_len, input_size): tensor containing the features of
#the input sequence. The input can also be a packed variable length sequence.
#h_0 of shape (num_layers, batch, hidden_size) for one direction RNN 
#tensor containing the initial hidden state for each element in the batch. 

# output of RNN layer are: output, h_n (n=1,2,...)
#output of shape (batch, seq_len, hidden_size): 
#tensor containing the output features (h_k) from the last layer of the RNN, for each k.                                      
#h_n (num_layers, batch, hidden_size): tensor containing the hidden state for k = seq_len.            

seqLen = 15
inputSize = 20
nSample = 100
hiddenSize = 10

#input is a sequence of input vectors
#for univarate time series, inputSize = 1
#for multivariate time series, inputSize > 1
#We can use more number of layers to increase the abstraction power 

rnnLayer = nn.RNN(
                input_size = inputSize,
                hidden_size = hiddenSize,
                num_layers = 1,
                batch_first= True
                )

X = torch.rand(nSample, seqLen, inputSize)
h_0 = torch.rand(1, nSample, hiddenSize)

Y, h_1 = rnnLayer(X, h_0)        
# Y.size()          torch.Size([100, 15, 10])  which is (batch, seq_len, hidden_size)
# h_1.size()        torch.Size([1, 100, 10])   which is (num_layers, batch, hidden_size)

# we can verify last Y out is equal to h_1
print(torch.all(torch.eq(Y[:,-1,:].squeeze(), h_1.squeeze())))


## We can also reproduce RNN layer by RNN cell

rnnCell = nn.RNNCell(
                input_size = inputSize,
                hidden_size = hiddenSize
                )

# initialize rnnCell paramters require the function of torch.nn.Parameter
rnnCell.weight_ih = torch.nn.Parameter(rnnLayer.state_dict()["weight_ih_l0"])
rnnCell.weight_hh = torch.nn.Parameter(rnnLayer.state_dict()["weight_hh_l0"])
rnnCell.bias_ih = torch.nn.Parameter(rnnLayer.state_dict()["bias_ih_l0"])
rnnCell.bias_hh = torch.nn.Parameter(rnnLayer.state_dict()["bias_hh_l0"])


# loop for sequence length
for i in range(X.shape[1]):
    if i == 0:
        hCell = rnnCell(X[:,i,:].squeeze(), h_0.squeeze())
    else:
        hCell = rnnCell(X[:,i,:].squeeze(), hCell)


# we can verify last hCell is equal to h_1
print(torch.all(torch.eq(hCell, h_1.squeeze())))
