import math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Any

from volcanoMDP import volcanoMDP
from utilRL import RLAlgorithm, simulate
random.seed(42)

############################################################
# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLgreedy(RLAlgorithm):
    def __init__(self, actions: Callable, discount: float, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb  # epsilon

        self.numIters = 0
        self.QValue = defaultdict(float)

    # Call this function to get the step size to update the Q values.
    def getStepSize(self) -> float:  # eta
        return 1.0 / math.sqrt(self.numIters)

    # Return the Q function - Q(s,a)
    def getQ(self, state: Tuple, action: Any) -> float:
        return self.QValue[(state, action)]

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state: Tuple) -> Any:
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max(((self.getQ(state, action), action) for action in self.actions(state)), key=lambda x: x[0])[1]

    # Return the V function - V(s)
    def getV(self, state):
        # BEGIN_YOUR_CODE
        Vopt = max(((self.getQ(state, action), action) for action in self.actions(state)), key=lambda x: x[0])[0]
        return Vopt
        # END_YOUR_CODE

    # We will call this function with (s, a, r, s'), which you should use to update the Q-values.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple) -> None:
        # BEGIN_YOUR_CODE
        qPredict = self.QValue[(state, action)]  # calculate q-predict
        if newState is not None:
            qTarget = reward + self.discount * self.getV(newState)
        else:
            qTarget = reward
        self.QValue[(state, action)] += self.getStepSize() * (qTarget - qPredict)

        # END_YOUR_CODE

    def printQs(self, stateSTART, states):
        print('\nQ-values', sorted(states))
        for act in sorted(self.actions(stateSTART)):
            print('{:7}'.format(act), end=' ')
            for st in sorted(states): print('%7.2f' % (self.getQ(st, act)), end=' ')
            print('')

        print('\nV-values', end='')
        for st in sorted(states): print('%7.2F' % (self.getV(st)), end=' ')
        print('')

        print('pi', end='      ')
        for st in sorted(states): print('%7s' % (max((self.getQ(st, act), act) for act in self.actions(st))[1]),
                                        end=' ')
        print('')


############################################################

if __name__ == '__main__':
    mdp = volcanoMDP(m=3, n=4, slipProb=0.1, gamma=0.8, r_lava=-50, r_fab=20, r_safe=2)
    mdp.computeStates()
    rl = QLgreedy(actions=mdp.actions, discount=mdp.discount(), explorationProb=0.5)  # explorationProb is epsilon

    nT = 100000
    totalRewards = simulate(mdp=mdp, rl=rl, numTrials=nT, maxIterations=1000, verbose=True, sort=False)

    rl.printQs(mdp.startState(), mdp.states)
    print('\nTotal trials ', nT, ', Total Iters ', rl.numIters, ', iterations-per-trial', rl.numIters / nT, end=' ')
    print(', Total Rewards ', sum(totalRewards), ', rewards-per-trial', sum(totalRewards) / nT)
