# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:34:18 2019

@author: BIN
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("threepointthree.json", encoding='utf-8') as f:
    dic = json.load(f)
    
hpreturn = []
diffreturn = []
volatility = []
for d in dic:
    hpreturn.append(dic[d]['hp_method']['Return'])
    diffreturn.append(dic[d]['diff_method']['Return'])
    volatility.append(dic[d]['Volatility'])


plt.figure(figsize=(10,4))
plt.subplot(121)
plt.hist(np.array(hpreturn))
plt.title('return')
plt.subplot(122)
plt.hist(np.array(volatility))
plt.title('volatility')

hpreturn = []
names = []
volatility = []
for d in dic:
    if dic[d]['hp_method']['Return']>0:
        names.append(d)
        hpreturn.append(dic[d]['hp_method']['Return'])
        volatility.append(dic[d]['Volatility'])