#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:21:34 2019

@author: yangyutu123
"""

# a good source: https://pymotw.com/2/multiprocessing/basics.html

import multiprocessing as mp
from multiprocessing import current_process


def worker():
    print("Hello, World! from " + current_process().name + "\n")


if __name__ == '__main__':
    nCpu = mp.cpu_count()
    print("total cpu count is " + str(nCpu) + "\n")
    jobs = []
    for i in range(nCpu):
        p = mp.Process(target=worker)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
