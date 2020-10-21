#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:21:34 2019

@author: yangyutu123
"""

# a good source: https://pymotw.com/2/multiprocessing/basics.html
# Here we show how to use Queue to store results from individual processes.

import torch.multiprocessing as mp

import torch
import time

# here d is the type code for double, 0 is the initial value.
# For the complete list of type codes, see https://docs.python.org/3/library/array.html


# summary:
# without lock, the final tenor matrix has different value for each entry
# using lock, the final tensor matrix will have the same value


# When there is no lock, we can see two process interleaved together, i.e., data racing.
def workerNoLock(sum, num):

    for i in range(10):
        time.sleep(0.1)
        # we need to use +=, otherwise will not work for example: sum = sum + num * torch.ones_like(sum)
        sum += num*torch.ones_like(sum)
        print(sum)

# When there is lock, we can see the correct summation performed by two process in order
def workerWithLock(lock, sum, num):

    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        sum += num*torch.ones_like(sum)
        print(sum)
    lock.release()


if __name__ == '__main__':
    sharedSum = torch.zeros(5, 5)

    # if we do not share memory, each process will have its own copy of sharedSum
    sharedSum.share_memory_()

    p1 = mp.Process(target=workerNoLock, args=(sharedSum, 1))
    p2 = mp.Process(target=workerNoLock, args=(sharedSum, 10))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(sharedSum)

    sharedSum.fill_(0)
    print("use lock")
    lock = mp.Lock()
    p1 = mp.Process(target=workerWithLock, args=(lock, sharedSum, 1))
    p2 = mp.Process(target=workerWithLock, args=(lock, sharedSum, 10))
    p1.start()
    p2.start()

    p1.join()
    p2.join()
    print(sharedSum)
