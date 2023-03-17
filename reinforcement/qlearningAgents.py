# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter() # A Counter is a dict with default 0


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state, action)]
        """util.raiseNotDefined()"""


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #Find the highest Q-value
        max_q_value = -1000000000
        for action in self.getLegalActions(state):
            q_value = self.getQValue(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
        #If there are legal actions, return found max q value
        if max_q_value > -1000000000:
            return max_q_value
        #If there are no legal actions, return 0.0
        else:
            return 0.0
        """util.raiseNotDefined()"""

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #If there are no legal actions, return None
        if not self.getLegalActions(state):
            return None

        best_action = ''
        max_q_value = -1000000000
        #Find the highest Q-value and its corresponding best action
        for action in self.getLegalActions(state):
            q_value = self.getQValue(state, action)
            if q_value > max_q_value:
                best_action = action
                max_q_value = q_value

        return best_action
        """util.raiseNotDefined()"""

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #Take a random action with probability self.epsilon
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        #Take the best policy action otherwise
        else:
            action = self.computeActionFromQValues(state)
            return action
        """util.raiseNotDefined()"""

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        old_q_value = self.getQValue(state, action)
        next_q_value = self.computeValueFromQValues(nextState)
        #Q(s,a) <- (1-alpha)*Q(s,a) + alpha*(reward + discount*max_a' Q(s', a'))
        self.values[(state, action)] = (1-self.alpha)*old_q_value + self.alpha*(reward + self.discount*next_q_value)
        """util.raiseNotDefined()"""

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        result = 0.0
        for index in self.featExtractor.getFeatures(state, action):
            #Q(state, action) = w_i*f_i(s,a)
            result += self.getWeights()[index]*self.featExtractor.getFeatures(state, action)[index]
        return result
        """util.raiseNotDefined()"""

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #difference = (reward + discount*max_a' Q(s', a')) - Q(s, a)
        difference = (reward + self.discount*self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        #update weights
        for index in self.featExtractor.getFeatures(state, action):
            #w_i = w_i + alpha*difference*f_i(s, a)
            self.getWeights()[index] += self.alpha*difference*self.featExtractor.getFeatures(state, action)[index]
        """util.raiseNotDefined()"""

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
