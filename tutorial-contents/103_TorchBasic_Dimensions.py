#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:30:35 2019

@author: yangyutu123
"""

import torch
import numpy as np

# create 2D torch tensor from a sequence
x = torch.arange(0,100).reshape((5, 20))

x.size()                              # return tuple-like object of dimensions
#torch.Size([5, 20])
      
x.transpose(0,1)                      # swaps the first dimension and the second dimension

x.view(4, 5, 5)                       # reshape the tensor to 3D tensor shape (4, 5, 5)


# use unsqueeze to add dimensionality
a = torch.randn(2,4)
# a.size() will give torch.Size([2, 4])
b = torch.unsqueeze(a,1)
# b.size() will give torch.Size([2, 1, 4])
b = torch.unsqueeze(a,0)
# b.size() will give torch.Size([1, 2, 4])

# use squeeze to remove dimensionality
# use unsqueeze to add dimensionality
a = torch.randn(1, 2,4)
b = torch.squeeze(a)
# b.size() will give torch.Size([2, 4])
