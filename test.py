#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:test.py
@time:2021/10/26
"""
import math
from collections import Counter
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([4, 3, 2, 1])
if __name__ == '__main__':
    print(math.sqrt(sum(np.square(x-y))))


