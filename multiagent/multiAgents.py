# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #if pacman is in the losing state, return a very negative value
        if currentGameState.isLose():
            return -1000000000
        #Compute the distance between the nearest food and the pacman
        nearest_food = None
        nearest_food_distance = 1000000000
        for food in newFood.asList():
            distance_to_food = util.manhattanDistance(newPos, food)
            if distance_to_food < nearest_food_distance:
                nearest_food = food
                nearest_food_distance = distance_to_food
        #Compute the distance between the nearest ghost and the pacman
        nearest_ghost = None
        nearest_ghost_distance = 1000000000
        for agentIndex in range(successorGameState.getNumAgents() - 1):
            ghost_pos = successorGameState.getGhostPosition(agentIndex + 1)
            distance_to_ghost = util.manhattanDistance(newPos, ghost_pos)
            if distance_to_ghost < nearest_ghost_distance:
                nearest_ghost = successorGameState.getGhostState(agentIndex + 1)
                nearest_ghost_distance = distance_to_ghost
        #if there are some food left
        if newFood.asList():
            #and the nearest ghost is at least 1 move away from the pacman
            if nearest_ghost_distance:
                #return the current score + 10*inverse of the distance to the nearest food - inverse of the distance to the nearest ghost
                #since larger those distances to the pacman, smaller impacts on the pacman.
                #10*inverse of the distance to the nearest food in order to emphasize the difference in weights between eating food and avoiding ghosts
                return successorGameState.getScore() + 10/nearest_food_distance - 1/nearest_ghost_distance
            #and the nearest ghost got you, return a very negative number as the game is at losing state
            else:
                return -1000000000
        #if there is no more food
        else:
            #and the nearest ghost is at least 1 move away from the pacman
            if nearest_ghost_distance:
                #return the current score - inverse of distance to the nearest ghost as closer the ghost more negative impact on pacman
                return successorGameState.getScore() - 1/nearest_ghost_distance
            #and the nearest ghost got ou, return a very negative number as the game is at losing state
            else:
                return -1000000000

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        number_of_agents = gameState.getNumAgents()
        number_of_moves = number_of_agents*depth
        (best_action, value) = self.min_max_Value(gameState, number_of_moves, 0)
        return best_action
    #def value(state):
    def min_max_Value(self, gameState, number_of_moves, agentIndex):
        evaluation = self.evaluationFunction
        #if the state is a terminal state: return the state's utility
        if number_of_moves == 0 or gameState.isWin():
            return (Directions.STOP, evaluation(gameState))
        elif gameState.isLose():
            return (Directions.STOP, evaluation(gameState))
        #if the next agent is MAX(pacman - agentIndex = 0): return max-value(state)
        if agentIndex == 0:
            return self.max_Value(gameState, number_of_moves, agentIndex)
        #if the next agent is MIN(agents index 1+): return min-value(state)
        else:
            return self.min_Value(gameState, number_of_moves, agentIndex)
    #def max-value(state):
    def max_Value(self, gameState, number_of_moves, agentIndex):
        #initialize v = -infinity
        max_action = Directions.STOP
        max_value = -1000000000
        #for each successor of state:
        for action in gameState.getLegalActions(agentIndex):
            successor_gameState = gameState.generateSuccessor(agentIndex, action)
            #value(successor)
            (successor_action, successor_value)  = self.min_max_Value(successor_gameState, number_of_moves - 1, 1)
            #v = max(v, value(successor))
            if max_value < successor_value:
                max_action = action
                max_value = successor_value
        #return v
        return (max_action, max_value)
    #def min-value(state):
    def min_Value(self, gameState, number_of_moves, agentIndex):
        #initialize v = +infinity
        min_action = Directions.STOP
        min_value = 1000000000
        #for each successor of state:
        for action in gameState.getLegalActions(agentIndex):
            successor_gameState = gameState.generateSuccessor(agentIndex, action)
            successor_agentIndex = agentIndex + 1
            number_of_agents = gameState.getNumAgents()
            #set agentIndex = 0 (pacman) when there is no more ghost
            if successor_agentIndex == number_of_agents:
                successor_agentIndex = 0
            #value(successor)
            (successor_action, successor_value) = self.min_max_Value(successor_gameState, number_of_moves - 1, successor_agentIndex)
            #v = min(v, value(successor))
            if min_value > successor_value:
                min_action = action
                min_value = successor_value
        #return v
        return (min_action, min_value)
        """util.raiseNotDefined()"""

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        number_of_agents = gameState.getNumAgents()
        number_of_moves = number_of_agents*depth
        #alpha: MAX's best option on path to root
        alpha = -1000000000
        #beta: MIN's best option on path to root
        beta = 1000000000
        (best_action, value) = self.alpha_beta_Value(gameState, number_of_moves, 0, alpha, beta)
        return best_action
    #def value(state, alpha, beta):
    def alpha_beta_Value(self, gameState, number_of_moves, agentIndex, alpha, beta):
        evaluation = self.evaluationFunction
        #if the state is a terminal state: return the state's utility
        if number_of_moves == 0 or gameState.isWin() or gameState.isLose():
            return (Directions.STOP, evaluation(gameState))
        #if the next agent is MAX: return max-value(state)
        if agentIndex == 0:
            return self.max_Value(gameState, number_of_moves, agentIndex, alpha, beta)
        #if the next agent is MIN: return min-value(state)
        else:
            return self.min_Value(gameState, number_of_moves, agentIndex, alpha, beta)
    #def max-value(state, alpha, beta):
    def max_Value(self, gameState, number_of_moves, agentIndex, alpha, beta):
        #initialize v = -infinity
        max_action = Directions.STOP
        max_value = -1000000000
        #for each successor of state:
        for action in gameState.getLegalActions(agentIndex):
            successor_gameState = gameState.generateSuccessor(agentIndex, action)
            #value(successor)
            (successor_action, successor_value)  = self.alpha_beta_Value(successor_gameState, number_of_moves - 1, 1, alpha, beta)
            #v = max(v, value(successor))
            if max_value < successor_value:
                max_action = action
                max_value = successor_value
            #if v > beta return v
            if max_value > beta:
                return (max_action, max_value)
            #alpha = max(a, v)
            if max_value > alpha:
                alpha = max_value
        #return v
        return (max_action, max_value)
    #def min-value(state, alpha, beta):
    def min_Value(self, gameState, number_of_moves, agentIndex, alpha, beta):
        #initialize v = +infinity
        min_action = Directions.STOP
        min_value = 1000000000
        #for each successor of state:
        for action in gameState.getLegalActions(agentIndex):
            successor_gameState = gameState.generateSuccessor(agentIndex, action)
            successor_agentIndex = agentIndex + 1
            number_of_agents = gameState.getNumAgents()
            #set agentIndex = 0 (pacman) when there is no more ghost
            if successor_agentIndex == number_of_agents:
                successor_agentIndex = 0
            #value(successor)
            (successor_action, successor_value) = self.alpha_beta_Value(successor_gameState, number_of_moves - 1, successor_agentIndex, alpha, beta)
            #v = min(v, value(successor))
            if min_value > successor_value:
                min_action = action
                min_value = successor_value
            #if v < alpha return v
            if min_value < alpha:
                return (min_action, min_value)
            #beta = min(beta, v)
            if min_value < beta:
                beta = min_value
        #return v
        return (min_action, min_value)
        """util.raiseNotDefined()"""

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        number_of_agents = gameState.getNumAgents()
        number_of_moves = number_of_agents*depth
        (best_action, best_value) = self.expectimax_Value(gameState, number_of_moves, 0)
        return best_action
    #def value(state):
    def expectimax_Value(self, gameState, number_of_moves, agentIndex):
        evaluation = self.evaluationFunction
        #if the state is a terminal state: return the state's utility
        if number_of_moves == 0 or gameState.isWin() or gameState.isLose():
            return (Directions.STOP, evaluation(gameState))
        #if the next agent is MAX: return max-value(state)
        if agentIndex == 0:
            return self.max_Value(gameState, number_of_moves, agentIndex)
        #if the next agent is EXP: return exp-value(state)
        else:
            return self.exp_Value(gameState, number_of_moves, agentIndex)
    #def max-value(state):
    def max_Value(self, gameState, number_of_moves, agentIndex):
        #initialize v = -infinity
        max_action = Directions.STOP
        max_value = -1000000000
        #for each successor of state:
        for action in gameState.getLegalActions(agentIndex):
            successor_gameState = gameState.generateSuccessor(agentIndex, action)
            #value(successor)
            (successor_action, successor_value)  = self.expectimax_Value(successor_gameState, number_of_moves - 1, 1)
            #v = max(v, value(successor))
            if max_value < successor_value:
                max_action = action
                max_value = successor_value
        #return v
        return (max_action, max_value)
    #def exp-value(state):
    def exp_Value(self, gameState, number_of_moves, agentIndex):
        #initialize v = 0
        expectimax_action = Directions.STOP
        expectimax_value = 0
        #for each successor of state:
        for action in gameState.getLegalActions(agentIndex):
            successor_gameState = gameState.generateSuccessor(agentIndex, action)
            successor_agentIndex = agentIndex + 1
            number_of_agents = gameState.getNumAgents()
            #if next agent should be pacman, set successor_agentIndex = 0
            if successor_agentIndex == number_of_agents:
                successor_agentIndex = 0
            #p = probability(successor)
            probability = 1 / len(gameState.getLegalActions(agentIndex)) #since adversary chooses actions uniformly at random
            #value(successor)
            (successor_action, successor_value) = self.expectimax_Value(successor_gameState, number_of_moves - 1, successor_agentIndex)
            #v += p*value(successor)
            expectimax_value += probability*successor_value
        #return v
        return (expectimax_action, expectimax_value)
        """util.raiseNotDefined()"""

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Calculate the distance to the nearest food, ghost and pellet,
    take inverse of the above values, then combine them with the currentState's
    score via linear combination and assignment of distinct weights. This is
    similar to my part 1 evaluation function, yet that lost 1 point on having
    less than 1000 average scores (957.5). Hence, I just add one more factor
    (pellet) to incrase the average scores.
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    current_food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]
    pellets = currentGameState.getCapsules()

    #if pacman is in the losing state, return a very negative value
    if currentGameState.isLose():
        return -1000000000
    #Compute the distance between the nearest food and the pacman
    nearest_food = None
    nearest_food_distance = 1000000000
    for food in current_food.asList():
        distance_to_food = util.manhattanDistance(pacman_pos, food)
        if distance_to_food < nearest_food_distance:
            nearest_food = food
            nearest_food_distance = distance_to_food
    #Compute the distance between the nearest ghost and the pacman
    nearest_ghost = None
    nearest_ghost_distance = 1000000000
    for agentIndex in range(currentGameState.getNumAgents() - 1):
        ghost_pos = currentGameState.getGhostPosition(agentIndex + 1)
        distance_to_ghost = util.manhattanDistance(pacman_pos, ghost_pos)
        if distance_to_ghost < nearest_ghost_distance:
            nearest_ghost = currentGameState.getGhostState(agentIndex + 1)
            nearest_ghost_distance = distance_to_ghost
    #Compute the distance between the nearest pellet and the pacman
    nearest_pellet = None
    nearest_pellet_distance = 1000000000
    for pellet in pellets:
        distance_to_pellet = util.manhattanDistance(pacman_pos, pellet)
        if distance_to_pellet < nearest_pellet_distance:
            nearest_pellet = pellet
            nearest_pellet_distance = distance_to_pellet
    #if there are some food left
    if current_food.asList():
        #and the nearest ghost is at least 1 move away from the pacman
        if nearest_ghost_distance:
            #and there is a pellet left
            if nearest_pellet_distance:
                #return the current score + 10*inverse of the distance to the nearest food - inverse of the distance to the nearest ghost + 100*inverse of the distance to the nearest pellet
                #since larger those distances to the pacman, smaller impacts on the pacman.
                #10*inverse of the distance to the nearest food in order to emphasize the difference in weights between eating food and avoiding ghosts
                #100*inverse of the distance to the nearest pellet further strengthes the importance on weights of having pellets for pacman
                return currentGameState.getScore() + 10/nearest_food_distance - 1/nearest_ghost_distance + 100/nearest_pellet_distance
            else:
                return currentGameState.getScore() + 10/nearest_food_distance - 1/nearest_ghost_distance
        else:
            return -1000000000
    else:
        if nearest_ghost_distance:
            if nearest_pellet_distance:
                return currentGameState.getScore() - 1/nearest_ghost_distance + 100/nearest_pellet_distance
            else:
                return currentGameState.getScore() - 1/nearest_ghost_distance
        else:
            return -1000000000

    """util.raiseNotDefined()"""

# Abbreviation
better = betterEvaluationFunction
