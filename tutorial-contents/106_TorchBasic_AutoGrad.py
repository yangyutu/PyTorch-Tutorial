#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:30:35 2019

@author: yangyutu123
"""

import torch

# We need to set requires_grad=True in order to get gradient with respect to x
x = torch.ones(2,2, requires_grad=True)
# construct function y as function of tensor x
y = torch.sum(torch.mm(x.t(),x))

# y take a derivative on all of its inputs
y.backward()
# get the gradient (y respect to x)
x.grad
#tensor([[4., 4.],
#        [4., 4.]])

# Another example
# A variable is used both as a leaf variable and an intermediate variable
# to get the grad, we have call a.retain_grad() before call backward()
a = torch.randn(2, 2, requires_grad = True)

a = ((a * 3) / (a - 1))

b = (a * a).sum()

a.retain_grad()

b.backward()

a.grad