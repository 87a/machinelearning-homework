#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:HW3.py
@time:2021/10/21
"""
import math
import numpy as np

X = np.array(
    [[10, 0],
     [30, 0],
     [10, 20],
     [20, 20]])
K = 2
# Mu = np.array(
#     [[20, 30],
#      [20, -10]])
Mu = np.array(
    [[0, 10],
     [30, 20]])
Z = np.zeros(X.shape[0])
EPOCHS = 100


def calcDist(x, mu):
    return math.sqrt((x[0] - mu[0]) ** 2 + (x[1] - mu[1]) ** 2)


if __name__ == '__main__':
    # print(calcDist(X[0], Mu[0]))
    for epoch in range(EPOCHS):
        Mu_ = np.zeros(Mu.shape)
        for i in range(0, len(X)):
            minMu = 0
            for k in range(K):
                dist = calcDist(X[i], Mu[k])
                if k == 0:
                    minDist = dist
                # print(dist)
                if dist < minDist:
                    minDist = dist
                    minMu = k
            Z[i] = minMu

        for k in range(K):
            xsum, ysum, count = 0, 0, 0
            for z in range(X.shape[0]):
                if Z[z] == k:
                    count += 1
                    xsum += X[z][0]
                    ysum += X[z][1]
            Mu_[k] = [xsum / count, ysum / count]
            c = (Mu_ == Mu)
            if c.all():
                break
            else:
                Mu = Mu_

    print(Z)
    print(Mu)
