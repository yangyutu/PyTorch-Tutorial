# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:50:21 2019

@author: yuguangyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#create an one-D cnn layer
#torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
in_channels = 2
out_channels = 3
kernel_size = 2
stride = 1
padding = 0

layer1D = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

# we can use layer.weight to see the weight parameter
# for each input channel output channel combination, we need a 2 element vector as the kernel
#layer.weight has torch.Size([3, 2, 2])
#Out[42]: 
#Parameter containing:
#tensor([[[-0.1135, -0.4443],
#         [ 0.1848, -0.0503]],
#
#        [[ 0.3654,  0.0368],
#         [ 0.1792,  0.4322]],
#
#        [[-0.0941, -0.2802],
#         [ 0.4187,  0.2692]]], requires_grad=True)
# we can use layer.bias to see the bias parameter
#layer.bias
#Out[41]: 
#Parameter containing:
#tensor([ 0.0148, -0.0342, -0.2400], requires_grad=True)


# The Conv1D layer expects these dimensions:
# (batchSize,  channels, length)
nSample = 100
length = 10

x = torch.rand((nSample, in_channels, length))
# y will be the output given by y_i = Wx_i + b
# y has size of (nSample, out_channels, length-1) 
y = layer1D(x)
# z the result of applying relu activation on y
z = F.relu(y)

#torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
maxPool1DLayer = nn.MaxPool1d(kernel_size = 2)

#z_out has size of torch.Size([100, 3, 4])
z_out = maxPool1DLayer(z)

#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
in_channels = 2
out_channels = 3
kernel_size = (2,3)
stride = 1
padding = 0
layer2D = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#layer2D.bias
#Out[44]: 
#Parameter containing:
#tensor([ 0.2738, -0.1455, -0.1591], requires_grad=True)

#layer2D.weight has size of [3, 2, 2, 3]
#for each input channel, 
#Out[45]: 
#Parameter containing:
#tensor([[[[ 0.0173, -0.1366, -0.0729],
#          [ 0.1766, -0.1864,  0.1083]],
#
#         [[ 0.2032, -0.0190,  0.2216],
#          [ 0.1381,  0.0468,  0.0930]]],
#
#
#        [[[-0.2332, -0.2527,  0.1362],
#          [-0.0390, -0.1201, -0.0666]],
#
#         [[ 0.0430,  0.1818, -0.1203],
#          [ 0.1490, -0.2199,  0.2033]]],
#
#
#        [[[ 0.1426,  0.0734, -0.1048],
#          [ 0.0098, -0.2640, -0.2624]],
#
#         [[ 0.1126, -0.2845,  0.2596],
#          [ 0.0796,  0.1327, -0.2879]]]], requires_grad=True)


# The Conv1D layer expects these dimensions:
# (batchSize,  channels, length)
nSample = 100
image_width = 16
image_height = 32

x = torch.rand((nSample, in_channels, image_width, image_height))
# y will be the output given by y_i = Wx_i + b
# y has size of (nSample, out_channels, length-1) 
y = layer2D(x)
# z the result of applying relu activation on y
# z has size torch.Size([100, 3, 15, 30])
z = F.relu(y)

#torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
maxPool2DLayer = nn.MaxPool2d(kernel_size = (2,2))

#z_out has size of torch.Size([100, 3, 7, 15])
z_out = maxPool2DLayer(z)
