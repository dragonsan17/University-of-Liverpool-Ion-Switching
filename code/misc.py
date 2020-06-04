# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:09:58 2020

@author: lenovo
"""

import numpy as np


a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])
c = []
for i in range(1,5):
    c = np.concatenate([c,a[i-2+1:i+1]])
    c = np.concatenate(([c,[b[i]]]))                    
    print(c)
    
print(c)