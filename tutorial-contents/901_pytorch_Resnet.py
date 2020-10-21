# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:25:41 2020

@author: yangy
"""

import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

#the first thing is to think about what we need. Well, first of all we need a 
# convolution layer and since PyTorch does not have the 'auto' padding in Conv2d, so we have to code ourself!

class Conv2Auto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
         # dynamic add padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

conv3x3 = partial(Conv2Auto, kernel_size=3, bias = False)

conv = conv3x3(in_channels=32, out_channels=64)
print(conv)
del conv


#The residual block takes an input with in_channels, applies some blocks of convolutional layers
# to reduce it to out_channels and sum it up to the original input. 

class ResNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion = 1, downsampling = 1, conv=conv3x3, *args, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv
        self.blocks = nn.Identity()
        self.shortcut = nn.Sequential(OrderedDict(
                {
                        'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size = 1, 
                                          stride = self.downsampling, bias = False),
                        'bn': nn.BatchNorm2d(self.expanded_channels)
                        }
                )) if self.should_apply_shortcut else None
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels
    
print(ResNetResidualBlock(32, 64))

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))
class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )                    
                    