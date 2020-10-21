#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:08:36 2019

@author: yangyutu123
"""

import json

jsonData = '{"a":1,"b":2,"c":3,"d":4,"e":5}';

text = json.loads(jsonData)
print(text)



# write to a file
with open('data.json','w') as outputFile:
    json.dump(text,outputFile)


# read from a file
with open('data.json','r') as inputFile:
    read = json.load(inputFile)