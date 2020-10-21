#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:24:03 2019

@author: yangyutu123
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:01:56 2019

@author: yangyutu123
"""

import time
import multiprocessing as mp
from multiprocessing import current_process
from termcolor import colored as clr


def run():
    print("Hello, World! from " + current_process().name + "\n")
    a = 0
    for i in range(100000):
        for j in range(1000):
            a += 1
    print(a)




if __name__ == "__main__":

    # we get number of agents greater than cpu counts, we will observe speed drop
    j = 20

    print(clr("Benchmark settings:", 'green'))
    print("No of cpu available: %d" % mp.cpu_count())
    print(clr("No of agents (processes): %d" % j))

    processes = [mp.Process(target=run) for p in range(j)]

    start = time.time()

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(clr("Time: %.3f seconds." % (time.time() - start), 'green'))