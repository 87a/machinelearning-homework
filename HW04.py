#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:HW04.py
@time:2021/11/05
"""

if __name__ == '__main__':
    Vend = 0
    Vin = 0
    for i in range(100):
        Vin = 1 / 3 * (4 + Vend) + 2 / 3 * (4 + Vin)
        print(Vin)
