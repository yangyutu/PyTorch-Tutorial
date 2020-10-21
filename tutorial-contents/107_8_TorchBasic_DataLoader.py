#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 00:28:23 2019

@author: yangyutu123
"""

import torch
import torch.utils.data as Data


BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

# we need to first convert to TensorDataset object
# TensorDataset is the dataset wrapper
# Each sample will be retrieved by indexing tensors along the first dimension.
torch_dataset = Data.TensorDataset(x,y)

# torch.utils.data.DataLoader is an iterator which provides function of
# Batching the data
# Shuffling the data
# Load the data in parallel using multiprocessing workers.


loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # shuffle data
    num_workers=2,              # use multi-threading to read data
)

nEpoch = 5

for epoch in range(nEpoch):
    for step, (batch_x, batch_y) in enumerate(loader):
        print(batch_x)
        print(batch_y)