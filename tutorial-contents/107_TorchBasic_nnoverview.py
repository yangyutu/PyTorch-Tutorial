# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:50:21 2019

@author: yuguangyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#create a fully connected layer from m to n units
n_feature = 3
n_out = 5
layer = nn.Linear(n_feature, n_out)

# we can use layer.weight to see the weight parameter
#layer.weight
#Out[30]: 
#Parameter containing:
#tensor([[-0.1094, -0.4106, -0.4935],
#        [ 0.4174,  0.0538,  0.2059],
#        [-0.4499, -0.5249, -0.4588],
#        [ 0.1829, -0.2622, -0.4352],
#        [ 0.5347,  0.0468, -0.2997]], requires_grad=True)
# we can use layer.bias to see the bias parameter
#layer.bias
#Out[29]: 
#Parameter containing:
#tensor([ 0.1998, -0.5683,  0.4021,  0.0524,  0.1726], requires_grad=True)

nSample = 100
x = torch.rand((nSample, n_feature))
# y will be the output given by y_i = Wx_i + b
# y has size of (nSample, n_out) 
y = layer(x)
# z the result of applying relu activation on y
z = F.relu(y)