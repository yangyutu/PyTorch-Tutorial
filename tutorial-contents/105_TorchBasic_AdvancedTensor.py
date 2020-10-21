#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:30:35 2019

@author: yangyutu123
"""

import torch
import numpy as np

# create torch tensor from list
a = torch.tensor([[1., -1.], [1., -1.]])
b = torch.rand([2,2])


# addition and subtraction
c = a + b

c = a - b


# tensor product
torch.mm(a, b)
c = a.mm(b)


