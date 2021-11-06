#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:volcanocrossing.py
@time:2021/11/05
"""
import random
import numpy as np

# hyper-parameters
# random.seed(42)
slipProb = 0.3  # probability of slipping which will cause random action
numIter = 100000  # number of iterations
gamma = 1  # discount rate


# actions
def E(state):
    newState = state + np.array([0, 1])
    if newState[1] > 3:
        newState = np.array(state)
    return newState.tolist()


def W(state):
    newState = state + np.array([0, -1])
    if newState[1] < 0:
        newState = np.array(state)
    return newState.tolist()


def S(state):
    newState = state + np.array([1, 0])
    if newState[0] > 2:
        newState = np.array(state)
    return newState.tolist()


def N(state):
    newState = state + np.array([-1, 0])
    if newState[0] < 0:
        newState = np.array(state)
    return newState.tolist()


class env(object):
    # initialize
    def __init__(self):
        self.rewards = np.array([[0, 0, -50, 20],
                                 [0, 0, -50, 0],
                                 [2, 0, 0, 0]])
        self.actions = [E, W, S, N]
        self.states = [[i, j] for i in range(3) for j in range(4)]
        # [[0, 0], [0, 1], [0, 2], [0, 3],
        #  [1, 0], [1, 1], [1, 2], [1, 3],
        #  [2, 0], [2, 1], [2, 2], [2, 3]]
        self.stateIndices = [i for i in range(12)]

    # judge isEnd
    def isEnd(self, stateIndex):
        return self.states[stateIndex] == [2, 0] or self.states[stateIndex] == [0, 3] \
               or self.states[stateIndex] == [0, 2] or self.states[stateIndex] == [1, 2]

    def succProbReward(self, stateIndex, action):
        # return list of [newState, prob, reward] triples
        result = []
        state = self.states[stateIndex]
        newState = action(state)
        slipState = self.actions[random.randint(0, len(self.actions) - 1)](state)
        newStateIndex = self.states.index(newState)
        slipStateIndex = self.states.index(slipState)
        result.append([newStateIndex, (1 - slipProb), self.rewards[newState[0], newState[1]]])
        result.append([slipStateIndex, slipProb, self.rewards[slipState[0], slipState[1]]])
        return result


def valueIteration(mdp):
    # initialize
    V = {}  # state -> Vopt[state]
    for stateIndex in mdp.stateIndices:
        V[stateIndex] = 0.

    def Q(stateIndex, action):
        return sum(prob * (reward + gamma * V[newStateIndex]) \
                   for newStateIndex, prob, reward in mdp.succProbReward(stateIndex, action))

    for i in range(numIter):
        print(f"iteration {i}")
        # compute the new values (newV) given the old values (V)
        newV = {}
        pi = {}
        for stateIndex in mdp.stateIndices:
            if mdp.isEnd(stateIndex):
                newV[stateIndex] = 0.
                pi[stateIndex] = 'none'
            else:
                Qs = []
                for action in mdp.actions:
                    Qs.append(Q(stateIndex, action))
                pi[stateIndex] = mdp.actions[Qs.index(max(Qs))].__name__
                newV[stateIndex] = max(Qs)

        # check for convergence
        if max(abs(V[stateIndex] - newV[stateIndex]) for stateIndex in mdp.stateIndices) < 1e-10:
            break
        V = newV

    # read out policy
    for i in range(3):
        for j in range(4):
            print('{:20} {:20}'.format(V[volcano.stateIndices[i * 4 + j]], pi[volcano.stateIndices[i * 4 + j]]), end='')
        print()


if __name__ == '__main__':
    volcano = env()
    valueIteration(mdp=volcano)

