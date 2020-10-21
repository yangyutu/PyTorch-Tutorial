#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:30:35 2019

@author: yangyutu123
"""

import torch
import numpy as np

x = torch.arange(12).reshape(4,3)

#tensor([[  0.,   1.,   2.],
#        [  3.,   4.,   5.],
#        [  6.,   7.,   8.],
#        [  9.,  10.,  11.]])

y = x[2:, :]
# note that x and y will share memory. If we change y, we will change x.
#tensor([[  6.,   7.,   8.],
#        [  9.,  10.,  11.]])

# local slicing
z = x[x > 5]
# tensor([ 6,  7,  8,  9, 10, 11])



# size of a tensor
x.size()
# number of elements
x.numel()

