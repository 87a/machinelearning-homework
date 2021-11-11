#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:volcano(qlearning).py
@time:2021/11/06
"""
import math
import random
import numpy as np

# hyper-parameters
random.seed(42)
slipProb = 0.1  # probability of slipping which will cause random action
numIter = 10000  # number of iterations
gamma = 0.8  # discount rate
epsilon = 0.5  # greedy
# eta = 0.01  # learning rate


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


# environment
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
        self.qtable = [[0 for j in range(len(self.actions))] for i in range(12)]
        # each state has 4 actions

    # judge isEnd
    def isEnd(self, stateIndex):
        return self.states[stateIndex] == [2, 0] or self.states[stateIndex] == [0, 3] \
               or self.states[stateIndex] == [0, 2] or self.states[stateIndex] == [1, 2]

    # return s' and reward according to s and a
    def takeAction(self, stateIndex, action):
        if random.random() > slipProb:  # not slip
            nextState = action(self.states[stateIndex])
            nextStateIndex = self.states.index(nextState)
            reward = self.rewards[nextState[0], nextState[1]]
        else:  # situation when slip
            nextState = self.actions[random.randint(0, len(self.actions) - 1)](self.states[stateIndex])
            nextStateIndex = self.states.index(nextState)
            reward = self.rewards[nextState[0], nextState[1]]
        return nextStateIndex, reward

    # choose action with epsilon greedy
    def chooseAction(self, stateIndex):
        stateActions = self.qtable[stateIndex]

        def judgeAllZero(stateActions):
            for stateAction in stateActions:
                if stateAction == 0:
                    continue
                else:
                    return False
            return True

        if random.random() > epsilon or judgeAllZero(stateActions):  # situation when choose random action
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        else:  # normal
            action = self.actions[stateActions.index(max(stateActions))]
        return action


# q-learning calculation
def qlearning(volcano):
    eta = 0
    for episode in range(numIter):  # main loop


        print(f"episode {episode}")
        stateIndex = 4  # start position
        isTerminated = False  # flag to determine when to stop
        while not isTerminated:
            eta += 1
            action = volcano.chooseAction(stateIndex=stateIndex)  # choose an action for current state
            nextStateIndex, reward = volcano.takeAction(stateIndex=stateIndex, action=action)  # get s' and reward
            actionIndex = volcano.actions.index(action)  # action's index
            qPredict = volcano.qtable[stateIndex][actionIndex]  # calculate q-predict
            if not volcano.isEnd(nextStateIndex):  # if not at end
                qTarget = reward + gamma * max(volcano.qtable[nextStateIndex])  # calculate q-target
            else:  # if at end
                qTarget = reward  # calculate q-target
                isTerminated = True  # modify flag to stop this epoch

            volcano.qtable[stateIndex][actionIndex] += (1.0 / math.sqrt(eta)) * (qTarget - qPredict)  # modify Q value
            stateIndex = nextStateIndex  # s = s'

    return volcano.qtable


if __name__ == '__main__':
    volcano = env()  # create env
    qtable = qlearning(volcano)
    for i in range(len(qtable)):
        if not volcano.isEnd(i):
            print('{:30} {:10}'.format(max(qtable[i]), volcano.actions[qtable[i].index(max(qtable[i]))].__name__),
                  end='')
        else:
            print('{:30} {:10}'.format(max(qtable[i]), 'none'),
                  end='')
        if (i + 1) % 4 == 0:
            print()
