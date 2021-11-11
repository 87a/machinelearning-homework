import os, collections
from typing import List, Tuple, Dict, Any


# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self) -> Tuple:
        raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state: Tuple) -> List[Any]:
        raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state: Tuple, action: Any) -> List[Tuple]:
        raise NotImplementedError("Override me")

    def discount(self):
        raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self, verbose=False):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                if verbose:
                    print(state, end=': ')
                    print(action, end=': ')
                    print(self.succAndProbReward(state, action))

                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        if verbose:
            print("%d states" % len(self.states))
            print(sorted(self.states))


############################################################

# An algorithm that solves an MDP (i.e., computes the optimal policy).
class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp: MDP): raise NotImplementedError("Override me")


############################################################

class ValueIteration(MDPAlgorithm):
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''

    def solve(self, mdp: MDP, epsilon=0.001, verbose=False):
        mdp.computeStates(verbose=verbose)

        def computeQ(mdp: MDP, V: Dict[Tuple, float], state: Tuple, action: Any) -> float:
            # Return Q(state, action) based on V(state).
            return sum(prob * (reward + mdp.discount() * V[newState]) \
                       for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp: MDP, V: Dict[Tuple, float]) -> Dict[Tuple, Any]:
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = collections.defaultdict(float)  # state -> value of state
        numIters = 0
        while True:
            newV = {}
            for state in mdp.states:
                # This evaluates to zero for end states, which have no available actions (by definition)
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))

            numIters += 1
            print(max(abs(V[state] - newV[state]) for state in mdp.states))
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

            # interim: Compute the optimal policy now
            if verbose:
                pi = computeOptimalPolicy(mdp, V)
                os.system('cls')  # clear
                print('Iteration {:15}'.format(numIters))
                print('s', 'pi(s)', 'V(s)', sep='\t')
                for state in sorted(mdp.states):
                    print(state, pi[state], V[state], sep='\t');

        # final: Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)

        if not verbose: print(("ValueIteration: %d iterations" % numIters))
        self.pi = pi
        self.V = V
############################################################
