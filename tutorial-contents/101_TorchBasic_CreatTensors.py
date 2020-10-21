#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:30:35 2019

@author: yangyutu123
"""

import torch
import numpy as np

# create torch tensor from list
torch.tensor([[1., -1.], [1., -1.]])

#tensor([[ 1., -1.],
#        [ 1., -1.]])

# create torch tensor from numpy as
torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
#tensor([[1, 2, 3],
#        [4, 5, 6]])


# create torch tensor with a specific type
# for the full list of different types, see https://pytorch.org/docs/stable/tensors.html
torch.tensor([[1., -1.], [1., -1.]],dtype = torch.int32)
#tensor([[ 1, -1],
#        [ 1, -1]], dtype=torch.int32)


# create torch tensor with special functions
torch.zeros([3, 4]) # creat a 3 by 4 tensor
torch.zeros([2, 4], dtype = torch.int32)
torch.ones([4, 4])


# Crate a tensor filled with random numbers from a uniform distribution from [0 1]
torch.rand([3, 4])
# Crate a tensor filled with random numbers from a standard normal distribution
torch.randn([3, 4])

# Crate a tensor like A filled with random numbers from a uniform distribution from [0 1]
A = torch.zeros([3, 4])
torch.rand_like(A)


# fill a tensor will constants
A.fill_(5)


# create an evenly-spaced one-dimensional tensor
torch.linspace(3, 10, steps=5)
torch.logspace(start=-10, end=10, steps=5)

# create a two-D tensor via reshape
torch.linspace(1, 100, steps = 20).reshape([-1, 20]) # change to 2D tensor

torch.arange(0,20).reshape([4,5])

torch.arange(0,30).reshape([-1,5]) # -1 means the system will figure out the dimensionality itself


# concatenate tensors
x =  torch.randn(4, 5)
#tensor([[-1.8204, -2.5838,  1.4017, -1.3317, -0.1914],
#        [ 1.4627,  0.2065,  1.3630, -0.7840, -1.5110],
#        [-0.9009,  1.7673,  1.2767, -1.3024,  0.3578],
#        [ 0.1862, -0.9444,  1.5803,  0.6879,  1.2083]])

torch.cat((x, x, x), dim = 0)
#tensor([[-1.8204, -2.5838,  1.4017, -1.3317, -0.1914],
#        [ 1.4627,  0.2065,  1.3630, -0.7840, -1.5110],
#        [-0.9009,  1.7673,  1.2767, -1.3024,  0.3578],
#        [ 0.1862, -0.9444,  1.5803,  0.6879,  1.2083],
#        [-1.8204, -2.5838,  1.4017, -1.3317, -0.1914],
#        [ 1.4627,  0.2065,  1.3630, -0.7840, -1.5110],
#        [-0.9009,  1.7673,  1.2767, -1.3024,  0.3578],
#        [ 0.1862, -0.9444,  1.5803,  0.6879,  1.2083],
#        [-1.8204, -2.5838,  1.4017, -1.3317, -0.1914],
#        [ 1.4627,  0.2065,  1.3630, -0.7840, -1.5110],
#        [-0.9009,  1.7673,  1.2767, -1.3024,  0.3578],
#        [ 0.1862, -0.9444,  1.5803,  0.6879,  1.2083]])
torch.cat((x, x), dim=1)
#tensor([[-1.8204, -2.5838,  1.4017, -1.3317, -0.1914, -1.8204, -2.5838,  1.4017,
#         -1.3317, -0.1914],
#        [ 1.4627,  0.2065,  1.3630, -0.7840, -1.5110,  1.4627,  0.2065,  1.3630,
#         -0.7840, -1.5110],
#        [-0.9009,  1.7673,  1.2767, -1.3024,  0.3578, -0.9009,  1.7673,  1.2767,
#         -1.3024,  0.3578],
#        [ 0.1862, -0.9444,  1.5803,  0.6879,  1.2083,  0.1862, -0.9444,  1.5803,
#          0.6879,  1.2083]])
