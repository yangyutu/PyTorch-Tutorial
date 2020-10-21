#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:21:34 2019

@author: yangyutu123
"""

# a good source: https://pymotw.com/2/multiprocessing/basics.html
# Here we show how to use Queue to store results from individual processes.

import multiprocessing as mp
import time

# here d is the type code for double, 0 is the initial value.
# For the complete list of type codes, see https://docs.python.org/3/library/array.html


# When there is no lock, we can see two process interleaved together, i.e., data racing.
def workerNoLock(sum, num):

    for i in range(10):
        time.sleep(0.1)
        sum.value = sum.value + num
        print(sum.value)

# When there is lock, we can see the correct summation performed by two process in order
def workerWithLock(lock, sum, num):

    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        sum.value = sum.value + num
        print(sum.value)
    lock.release()
if __name__ == '__main__':
    sharedSum = mp.Value('d', 0)

    p1 = mp.Process(target=workerNoLock, args=(sharedSum, 1))
    p2 = mp.Process(target=workerNoLock, args=(sharedSum, 10))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("use lock")
    lock = mp.Lock()
    p1 = mp.Process(target=workerWithLock, args=(lock, sharedSum, 1))
    p2 = mp.Process(target=workerWithLock, args=(lock, sharedSum, 10))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

