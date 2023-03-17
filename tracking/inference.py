# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        #Get the sum of values for all keys
        sum = self.total()
        #If the sum of values are not equal to zero
        if sum != 0:
            #Iterate over key in keys
            for key in self.keys():
                #Divide the value by the sum of total values
                self[key] = self.__getitem__(key)/sum
        """raiseNotDefined()"""

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        #Get a random value
        random_value = random.random()
        #Normalize the distribution before further computation
        self.normalize()
        #Sort the distribution with some increasing orders
        distribution = sorted(self.items())
        sum = 0
        #Iterate over the distribution
        for key, value in distribution:
            #Find the corresponding key to random value through accumulating sum of distributions
            if random_value >= sum and random_value < sum + value:
                return key
            sum += value
        """raiseNotDefined()"""

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        #Special case: if the ghost's position is the jail position
        if ghostPosition == jailPosition:
            #then the observation is None with probability 1 -> the ghost is in jail with probability 1
            if noisyDistance == None:
                return 1
            #if the distance reading is not None, then the ghost is in jail with probability 0
            else:
                return 0
        #Find the true distance between Pacman's location and the ghost's location
        trueDistance = manhattanDistance(pacmanPosition, ghostPosition)
        #if the distance sensor gets nothing
        if noisyDistance == None:
            #P(noisyDistance | trueDistance) = P(0 | pacmanPosition, ghostPosition) = 0
            return 0
        else:
            #P(noisyDistance | trueDistance) = P(noisyDistance | pacmanPosition, ghostPosition)
            return busters.getObservationProbability(noisyDistance, trueDistance)
        """raiseNotDefined()"""

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        #Iterate over legal positions plus the special jail position
        for ghost_position in self.allPositions:
            pacman_position = gameState.getPacmanPosition()
            jail_position = self.getJailPosition()
            #Update the belief at every position on the map after receiving a sensor reaing
            self.beliefs[ghost_position] *= self.getObservationProb(observation, pacman_position, ghost_position, jail_position)
        """raiseNotDefined()"""

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        #Create an empty discrete distribution
        new_beliefs = DiscreteDistribution()
        #Iterate over previous ghost positions at time t
        for oldPos in self.allPositions:
            #Distribution over new positions for the ghost at time t+1
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            #Iterate over new ghost positions
            for p in self.allPositions:
                #Update the belief at every position p with the probability that the ghost is at position p at time t+1 multiplied by the belief of ghost old position at time t
                new_beliefs[p] += newPosDist[p]*self.beliefs[oldPos]
        #Update old beliefs with new beliefs
        self.beliefs = new_beliefs
        """raiseNotDefined()"""

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        #Compute number of particles left for each legal board position if distributed uniformly
        num = self.numParticles/len(self.legalPositions)
        index = 0
        #If there are more particles than number of legal board position
        while index < num:
            #locate one particle in each legal board position
            self.particles += self.legalPositions
            index += 1
        index = 0
        #If there are less particles left for each legal board position, locate the rest particles in the legal board position evenly
        while index < self.numParticles-num*len(self.legalPositions):
            self.particles += self.legalPositions[index]
            index += 1
        """raiseNotDefined()"""

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        old_beliefs = self.getBeliefDistribution()
        #Iterate over particles in old beliefs
        for particle in old_beliefs:
            pacman_position = gameState.getPacmanPosition()
            jail_position = self.getJailPosition()
            #Update the weight of a particle with the probability of an observation given Pacnman's position, a potential ghost position, and the jail position
            old_beliefs[particle] *= self.getObservationProb(observation, pacman_position, particle, jail_position)
        #Special case: When all particles receive zero weight
        if old_beliefs.total() == 0:
            #the list of particles should be reinitialized by calling initializeUniformly
            self.initializeUniformly(gameState)
            return
        #Create a new list of particles
        new_particles = []
        index = 0
        #Resample from the updated weighted distributions to construct new list of particles
        while index < self.numParticles:
            new_particles.append(old_beliefs.sample())
            index += 1
        #Update old particle list to new particle list
        self.particles = new_particles
        """raiseNotDefined()"""


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        #Create a new list of particles
        new_particles = []
        #Iterate over particles in old position
        for oldPos in self.particles:
            #Obtain the distribution over new positions for ghosts, given its previous position
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            #Sample each particle's next state and append this new position to the new list of particles
            new_particles.append(newPosDist.sample())
        #Assign the new list of particles back to self.particles
        self.particles = new_particles
        """raiseNotDefined()"""

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.

        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        #Create an empty belief distribution
        belief_distribution = DiscreteDistribution()
        #Iterate over the list of particles
        for particle in self.particles:
            #Increase the counter of corresponding number of particles by 1
            belief_distribution[particle] += 1
        #Normalize the distribution
        belief_distribution.normalize()
        return belief_distribution
        """raiseNotDefined()"""


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        #Compute the list of permutations from the Cartesian products of ghost positions can be occupied
        permutation = list(itertools.product(self.legalPositions, repeat = self.numGhosts))
        #Shuffle the list of permutations obtained
        random.shuffle(permutation)
        #Compute number of particles left for each legal board position if distributed uniformly
        num = self.numParticles/len(permutation)
        index = 0
        #If there are more particles than number of legal board position
        while index < num:
            #locate one particle in each legal board position
            self.particles += permutation
            index += 1
        index2 = 0
        #If there are less particles left for each legal board position, locate the rest particles in the legal board position evenly
        while index2 < self.numParticles-index*len(permutation):
            self.particles += permutation[index]
            index2 += 1
        """raiseNotDefined()"""

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        old_beliefs = self.getBeliefDistribution()
        #Iterate over particles in old beliefs
        for particles in old_beliefs:
            pacman_position = gameState.getPacmanPosition()
            #Compute list of jail positions for multiple ghosts
            jail_position_list = []
            for i in range(self.numGhosts):
                jail_position_list.append(self.getJailPosition(i))
            #Iterate over number of ghosts
            for i in range(self.numGhosts):
                #Update the beliefs with the probability of an observation given Pacman's position, a potential ghost position, and the jail position
                old_beliefs[particles] *= self.getObservationProb(observation[i], pacman_position, particles[i], jail_position_list[i])
        #Special case: When all particles receive zero weight
        if old_beliefs.total() == 0:
            #the list of particles should be reinitialized by calling initializeUniformly
            self.initializeUniformly(gameState)
            return
        #Create a new list of particles
        new_particles = []
        index = 0
        #Resample from the updated weighted distributions to construct new list of particles
        while index < self.numParticles:
            new_particles.append(old_beliefs.sample())
            index += 1
        #Update old particle list to new particle list
        self.particles = new_particles
        """raiseNotDefined()"""

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            #Loop over the ghosts
            for i in range(self.numGhosts):
                #Obtain the distribution over new positions for the single ghost, given the list of previous positions of all the ghosts
                newPosDist = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i])
                #Draw a new position from the above distribution and update the corresponding entry in newParticle
                newParticle[i] = newPosDist.sample()
            """raiseNotDefined()"""

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
