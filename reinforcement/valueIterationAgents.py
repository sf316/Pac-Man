# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        index = 0
        #Iterate self.iterations times
        while index < self.iterations:
            values = util.Counter()
            for state in self.mdp.getStates():
                #find the max q value for current state
                max_q_value = -1000000000
                for action in self.mdp.getPossibleActions(state):
                    #compute q value with given action
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
                #if there are legal actions, update state's value
                if max_q_value > -1000000000:
                    values[state] = max_q_value
            #update states' values
            self.values = values
            index += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        for transition_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            #R(s,a,s')
            reward = self.mdp.getReward(state, action, transition_state)
            #V_k(s')
            optimal_value_k_transition_state = self.values[transition_state]
            #sum over s' on T(s,a,s')*[R(s,a,s') + gamma*V_k(s')]
            q_value = q_value + prob*(reward + self.discount*optimal_value_k_transition_state)

        return q_value
        """util.raiseNotDefined()"""

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #If there are no legal actions, return None
        if self.mdp.isTerminal(state):
            return None
        best_action = ''
        max_q_value = -1000000000
        #find the max q value and its corresponding action as the best action
        for action in self.mdp.getPossibleActions(state):
            #compute q value with given action
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_q_value:
                best_action = action
                max_q_value = q_value

        return best_action
        """util.raiseNotDefined()"""

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        index = 0
        #Iterate self.iterations times
        while index < self.iterations:
            #update the available state one by one
            num_of_states = len(self.mdp.getStates())
            state = self.mdp.getStates()[index % num_of_states]
            #find the max q value for current state
            max_q_value = -1000000000
            for action in self.mdp.getPossibleActions(state):
                #compute q value with given action
                q_value = self.computeQValueFromValues(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
            #if there are legal actions, update state's value
            if max_q_value > -1000000000:
                self.values[state] = max_q_value
            index += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        def computeMaxQValue(state):
            #find the max q value for current state
            max_q_value = -1000000000
            for action in self.mdp.getPossibleActions(state):
                #compute q value with given action
                q_value = self.computeQValueFromValues(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
            return max_q_value

        #Compute predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for transition_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    #add transition_state to predecessors if it's not in the directionary
                    if transition_state not in predecessors:
                        predecessors[transition_state] = [state]
                    #append the state to the list of predecessors of transition_state
                    else:
                        predecessors[transition_state].append(state)
        #Initialize an empty prioirty queue
        priority_queue = util.PriorityQueue()
        #For each non-terminal state s, do
        for state in self.mdp.getStates():
            #skip terminal states
            if self.mdp.isTerminal(state):
                continue
            #Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s
            difference = abs(self.values[state] - computeMaxQValue(state))
            #Push s into the priority queue with priority -diff
            priority_queue.update(state, -difference)
        #For iteration in 0,1,2,...,self.iterations -1 do
        index = 0
        while index < self.iterations:
            #If the priority queue is empty, then terminate
            if priority_queue.isEmpty():
                return
            #Pop a state s off the priority queue
            state = priority_queue.pop()
            #Update the value of s(it it is not a terminal state) in self.values
            self.values[state] = computeMaxQValue(state)
            #For each predecessor p of s, do:
            for predecessor in predecessors[state]:
                #Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p
                difference = abs(self.values[predecessor] - computeMaxQValue(predecessor))
                #if diff > theta, push p into the priority queue with priority -diff
                if difference > self.theta:
                    priority_queue.update(predecessor, -difference)
            index += 1
