#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:01:56 2019

@author: yangyutu123
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process
from termcolor import colored as clr


class TestNet(nn.Module):
    """ Net for low-dimensional games.
    """
    def __init__(self, input_channels, hist_len, action_no):
        self.in_channels = hist_len * input_channels

        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.lin1 = nn.Linear(512, 32)
        self.head = nn.Linear(32, action_no)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))


def play(steps, model, allocate_on_cpu=False):
    state = torch.rand(1, 1, 24, 24)
    batch = torch.rand(5, 1, 24, 24)

    model(state).data.max(1)[1]

    model.zero_grad()
    y = model(batch).max(1)[0]
    t = torch.randn(y.size())
    loss = F.smooth_l1_loss(y, t)
    loss.backward()


def work_unit(pidx, steps, model):
    # this line is critical !!!!, if not set, there will be multithread in each indivial process, will greatly reduce speed.
    torch.set_num_threads(1)
    print("[worker #%d] started." % pidx)
    print("[worker #%d] has %d threads." % (pidx, torch.get_num_threads()))

    for i in range(steps):
        play(steps, model)

    print("[worker #%d] finished." % pidx)


if __name__ == "__main__":

    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.set_default_tensor_type("torch.FloatTensor")

    steps = 30000
    j = 8

    print(clr("Benchmark settings:", 'green'))
    print("No of threads available: %d" % torch.get_num_threads())
    print(clr("No of 'game steps': %d" % steps))
    print(clr("No of agents (processes): %d" % j))

    model = TestNet(1, 1, 3).share_memory()
    p_steps = int(steps / j)

    processes = [Process(target=work_unit, args=(p, p_steps, model))
                 for p in range(j)]

    start = time.time()

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(clr("Time: %.3f seconds." % (time.time() - start), 'green'))