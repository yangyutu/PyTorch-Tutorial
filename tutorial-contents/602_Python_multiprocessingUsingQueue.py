#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:21:34 2019

@author: yangyutu123
"""

# a good source: https://pymotw.com/2/multiprocessing/basics.html
# Here we show how to use Queue to store results from individual processes.

import multiprocessing as mp


def worker(q):
    res = 0
    for i in range(100):
        res += i
    q.put(res)



if __name__ == '__main__':
    q = mp.Queue()
    jobs = []
    for i in range(5):
        p = mp.Process(target=worker, args=(q,))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    totalSum = 0
    count = 0
    while not q.empty():
        res = q.get()
        totalSum += res
    print(totalSum)