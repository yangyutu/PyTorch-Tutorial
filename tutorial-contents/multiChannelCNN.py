#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:31:45 2019

@author: yangyutu123
"""

import torch
import torch.nn as nn
import os
torch.manual_seed(1)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, inputWdith, num_action):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential( # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(1,             # input channel
                      16,            # output channel
                      kernel_size=2, # filter size
                      stride=1,
                      padding=1),   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # inputWdith / 2
        # add a fully connected layer
        width = int(inputWdith / 4) + 1
        self.fc = nn.Linear(width * width * 32 + 2, num_action)

    def forward(self, state):
        x = state['sensor']
        y = state['target']
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        out = torch.cat((xout, y),1)
        out = self.fc(out)
        return out
    
cnn = ConvNet(11, 4)
x = torch.rand([10, 1, 11, 11])
y = torch.rand([10, 2])
input = {'sensor':x, 'target':y}

predict = cnn(input) 