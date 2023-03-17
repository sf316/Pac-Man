# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    """ initial the search tree using the initial state of problem"""
    start = problem.getStartState()
    visited = []
    stack = util.Stack()
    stack.push((start, []))

    #loop do
    while True:
        #if there are no candidates for expansion then return failure
        if stack.isEmpty():
            return None
        #choose a leaf node for expansion according to strategy(dfs)
        pair = stack.pop()
        node = pair[0]
        path_to_node = pair[1]
        visited.append(node)
        if problem.isGoalState(node):
            return path_to_node
        #else expand the node and add the resulting nodes to the search tree
        for successor in problem.getSuccessors(node):
            new_node = successor[0]
            path_to_new_node = [successor[1]]

            if new_node not in visited:
                stack.push((new_node, path_to_node + path_to_new_node))

    return None
    """util.raiseNotDefined()"""

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    visited = []
    queue = util.Queue()
    queue.push((start, []))
    visited.append(start)

    #loop do
    while True:
        #if there are no candidates for expansion then return failure
        if queue.isEmpty():
            return None
        #choose a leaf node for expansion according to strategy(bfs)
        pair = queue.pop()
        node = pair[0]
        path_to_node = pair[1]
        if problem.isGoalState(node):
            return path_to_node
        #else expand the node and add the resulting nodes to the search tree
        for successor in problem.getSuccessors(node):
            new_node = successor[0]
            path_to_new_node = [successor[1]]

            if new_node not in visited:
                queue.push((new_node, path_to_node + path_to_new_node))
                visited.append(new_node)

    return None
    """util.raiseNotDefined()"""


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    closed = []
    fringe = util.PriorityQueue()
    fringe.push((start, [], 0), 0)

    #loop do
    while True:
        #if fringe is empty then return failure
        if fringe.isEmpty():
            return None
        #node <- REMOVE-FRONT(fringe)
        pair = fringe.pop()
        node = pair[0]
        path_to_node = pair[1]
        cost = pair[2]
        #if GOAL-TEST(problem, STATE[node]) then return node
        if problem.isGoalState(node):
            return path_to_node
        #if STATE[node] is not in closed then
        if node not in closed:
            #add STATE[node] is not in closed then
            closed.append(node)
            for successor in problem.getSuccessors(node):
                new_node = successor[0]
                path_to_new_node = [successor[1]]
                new_node_cost = successor[2]
                #fringe<-INSERT(child-node, fringe)
                fringe.update((new_node, path_to_node + path_to_new_node, cost + new_node_cost), cost + new_node_cost)

    return None
    """util.raiseNotDefined()"""

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    closed = []
    fringe = util.PriorityQueue()
    fringe.push((start, [], 0), heuristic(start, problem))

    #loop do
    while True:
        #if fringe is empty then return failure
        if fringe.isEmpty():
            return None
        #node <- REMOVE-FRONT(fringe)
        pair = fringe.pop()
        node = pair[0]
        path_to_node = pair[1]
        cost = pair[2]
        #if GOAL-TEST(problem, STATE[node]) then return node
        if problem.isGoalState(node):
            return path_to_node
        #if STATE[node] is not in closed then
        if node not in closed:
            #add STATE[node] is not in closed then
            closed.append(node)
            #expand the node and add the resulting nodes to the search tree
            for successor in problem.getSuccessors(node):
                new_node = successor[0]
                path_to_new_node = [successor[1]]
                new_node_cost = successor[2]
                #fringe<-INSERT(child-node, fringe)
                fringe.update((new_node, path_to_node + path_to_new_node, cost + new_node_cost), cost + new_node_cost + heuristic(new_node, problem))

    return None
    """util.raiseNotDefined()"""


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
