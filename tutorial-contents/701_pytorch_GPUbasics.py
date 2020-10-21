# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:57:20 2019

@author: yangyutu123
"""

import torch
import torch.nn as nn
import torch.utils.data as Data


# count number of GPU device
print('number of GPU devices:' + str(torch.cuda.device_count()))


# check if GPU is available
print('check GPU availability:' + str(torch.cuda.is_available()))


# create a tensor at CPU

t1 = torch.rand(4,4)
print(t1)
#tensor([[0.3852, 0.0048, 0.4986, 0.4301],
#        [0.5420, 0.7559, 0.9213, 0.8685],
#        [0.3394, 0.2820, 0.3592, 0.1967],
#        [0.4528, 0.1827, 0.9740, 0.1339]])

# move a cpu tenor to GPU tensor
# note that this is deep copy: change t1_cuda will not change t1
t1_cuda = t1.cuda()
print(t1_cuda)
#tensor([[0.3852, 0.0048, 0.4986, 0.4301],
#        [0.5420, 0.7559, 0.9213, 0.8685],
#        [0.3394, 0.2820, 0.3592, 0.1967],
#        [0.4528, 0.1827, 0.9740, 0.1339]], device='cuda:0')

# check if a tenor is at GPU
print(t1_cuda.is_cuda)


# directly create a tensor at GPU

t2 = torch.rand(4,4, device = torch.device("cuda"))
#tensor([[0.2293, 0.5769, 0.5550, 0.5157],
#        [0.5370, 0.0638, 0.1560, 0.4257],
#        [0.4223, 0.7144, 0.7721, 0.7946],
#        [0.1446, 0.1694, 0.5850, 0.5443]], device='cuda:0')

# note that tensor at different devices cannot directly compute
# because they are belonging to different types: torch.FloatTensor Vs. torch.cuda.FloatTensor
t1 + t2

# move GPU tenor to CPU
t2_cpu = t2.cpu()


# note that only CPU tensor can be converted to numpy
# t2.data.numpy() will throw error
t2_cpu.data.numpy()