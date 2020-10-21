# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:50:21 2019

@author: yuguangyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ordinary dropout
#During training, randomly zeroes some of the elements of the input tensor with probability 
#p using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
m = nn.Dropout(p=0.2)
x = torch.randn(20,16)
# dropout will apply to x
output = m(x)




# 2D dropout, apply to convolution layer output
# Randomly zero out entire channels. Each channel will be zeroed out independently 
# on every forward call. with probability p using samples from a Bernoulli distribution.
# the input should be (NSample, NChannel, H, W)
# the output will be (NSample, NChannel, H, W)
m = nn.Dropout2d(p=0.2)
x = torch.randn(20,16, 32, 32)
# dropout will apply to x
output = m(x)


