import random
from typing import Tuple
import random

from utilMDP import MDP, ValueIteration


class volcanoMDP(MDP):
    # m and n are the numbers of rows and columns, respectively.
    def __init__(self, m=3, n=4, slipProb=0.1, gamma: float = 1, r_lava=-50, r_fab=20, r_safe=2):
        self.m = m
        self.n = n
        self.slipProb = slipProb
        self.gamma = gamma
        self.r_lava = r_lava
        self.r_fabulous = r_fab
        self.r_safeboring = r_safe

    def discount(self) -> float:
        return self.gamma

    # Return the start state. Any state is represented by a tuple of row and column, so the start state is (2,1)
    def startState(self):
        return (2, 1)

    # BEGIN_YOUR_CODE
    # Any auxiliary functions supporting the actions and succAndProbReward functions
    def isEnd(self, state: Tuple):
        return state == (3, 1) or state == (1, 3) or state == (1, 4) or state == (2, 3)

    def reward(self, state: Tuple) -> int:
        if state == (3, 1):
            return self.r_safeboring
        if state == (1, 3) or state == (2, 3):
            return self.r_lava
        if state == (1, 4):
            return self.r_fabulous
        return 0

    def takeAction(self, state: Tuple, action) -> Tuple:
        if action == 'E':
            newStateY = state[1] + 1 if state[1] < 4 else state[1]
            newState = (state[0], newStateY)
            return newState
        if action == 'W':
            newStateY = state[1] - 1 if state[1] > 1 else state[1]
            newState = (state[0], newStateY)
            return newState
        if action == 'S':
            newStateX = state[0] + 1 if state[0] < 3 else state[0]
            newState = (newStateX, state[1])
            return newState
        if action == 'N':
            newStateX = state[0] - 1 if state[0] > 1 else state[0]
            newState = (newStateX, state[1])
            return newState

    # END_YOUR_CODE

    # Return set of actions possible from a state. Each action can be represented by a letter, for example, E stands for moving to East
    def actions(self, state):
        # BEGIN_YOUR_CODE
        result = []
        if not self.isEnd(state):
            result = ['W', 'E', 'S', 'N']
            return result
        result = ['done']
        return result

    # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges coming out of a chance node (state, action).
    # Mapping to notation from the lecture notes:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action):
        lstSucc = list()
        # BEGIN_YOUR_CODE
        if not self.isEnd(state):
            newState = self.takeAction(state, action)
            lstSucc.append(
                (newState, 1 - self.slipProb + self.slipProb / len(self.actions(state)), self.reward(newState)))
            actionIndex = self.actions(state).index(action)
            otherActions = [i for i in range(len(self.actions(state))) if i != actionIndex]
            otherNewStates = [self.takeAction(state, self.actions(state)[i]) for i in otherActions]
            for i in otherNewStates:
                lstSucc.append((i, self.slipProb / 4, self.reward(i)))

        # END_YOUR_CODE
        return lstSucc


if __name__ == '__main__':
    mdp = volcanoMDP(m=3, n=4, slipProb=0.1, gamma=0.8, r_lava=-50, r_fab=20, r_safe=2)
    # print(mdp.takeAction((3, 4), 'W'))
    # print(mdp.succAndProbReward((2, 2), 'E'))
    # print(mdp.actions((3, 4)))
    vi = ValueIteration()

    vi.solve(mdp=mdp, epsilon=0.00001, verbose=True)

    print('s', 'pi(s)', 'V(s)', sep='\t')
    for state in sorted(mdp.states): print(state, vi.pi[state], vi.V[state], sep='\t');
