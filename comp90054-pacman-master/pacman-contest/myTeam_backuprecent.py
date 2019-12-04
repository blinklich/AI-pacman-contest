# myTeam_backup.py
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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions, Actions
from util import nearestPoint
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
ATTACK = 0
RUN = 1
MAXCARRY = 1


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        midX = gameState.data.layout.width / 2
        if gameState.isOnRedTeam(self.index):
            midX = midX - 1
        else:
            midX = midX + 1

        legalPositions = [legalPos for legalPos in gameState.getWalls().asList(False)]

        self.start = gameState.getAgentPosition(self.index)
        self.beforeFood = self.getFoodYouAreDefending(gameState).asList()
        self.border = [pos for pos in legalPositions if pos[0] == midX]
        self.alpha = 0.5
        self.gama = 0.5
        self.farPoint = (0, 0)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        return invaders

    def getGhost(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]

        return ghosts

    def getScaredGhost(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 5]

        return ghosts

    def getMinDistanceToMid(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        return min([self.getMazeDistance(myPos, borderPos) for borderPos in self.border])


class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index):
        # Agent index for querying state
        ReflexCaptureAgent.__init__(self, index)

        self.lastFood = 0
        self.carryScore = 0
        self.lastScore = -1
        self.state = ATTACK

        # pust some init weights
        self.weights = {'successorScore': 30,
                        'distanceToFood': -100,
                        'getCap': -50,
                        'reverse': -5,
                        'back': 30,
                        'ghost': -500,
                        'scaredGhost': 10}

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        value = 0.0
        for feature in features:
            product = features[feature] * self.weights[feature]
            value += product
        return value

    def getMaxStateAction(self, gameState):
        qValues = []
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        if len(actions) == 0:
            return None
        else:
            return max((self.getQValue(gameState, action), action) for action in actions)

    def update(self, gameState, action):
        weights = self.weights
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)
        maxStateAction = self.getMaxStateAction(nextState)
        myPos = gameState.getAgentState(self.index).getPosition()

        ghosts = self.getGhost(gameState)
        ghostPos = [ghost.getPosition() for ghost in ghosts]
        minDistance = 99999
        if len(ghostPos) > 0:
            minDistance = (min(self.getMazeDistance(myPos, pos) for pos in ghostPos))

        # set reward
        if minDistance < 3:
            reward = -1
        else:
            reward = 1
            
        if self.getScore(nextState) - self.getScore(gameState) <= 0:
            reward -= 1
        else:
            reward += 1

        for feature in features:
            weights[feature] += self.alpha * features[feature] * (
                    reward + self.gama * maxStateAction[0] - weights[feature])

        self.weights = weights

    def chooseAction(self, gameState):
        actions = self.getMaxStateAction(gameState)
        self.update(gameState, actions[1])

        action = actions[1]
        return action

    def getFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPosition = successor.getAgentState(self.index).getPosition()
        walls = gameState.getWalls()

        weight = 10 * walls.width * walls.height

        score = self.getScore(gameState)
        remainFood = len(self.getFood(gameState).asList())
        if score > self.lastScore:
            self.lastScore = score
            self.carryScore = 0
        # if get food, carryScore++
        if remainFood < self.lastFood:
            self.carryScore += (self.lastFood - remainFood)
            self.lastFood = remainFood
        # dead so need to init
        elif remainFood > self.lastFood:
            self.carryScore = 0
            self.lastFood = remainFood
            self.state = ATTACK

        # if get enough, run
        if self.state == ATTACK and self.carryScore >= MAXCARRY:
            self.state = RUN
        # when back home, attack again
        elif self.state == RUN and not gameState.getAgentState(self.index).isPacman:
            self.state = ATTACK
            self.carryScore = 0

        # ghost
        ghosts = self.getGhost(gameState)
        ghostPos = [ghost.getPosition() for ghost in ghosts]
        if len(ghostPos) > 0:
            minGost = min(self.getMazeDistance(myPosition, pos) for pos in ghostPos)
            features['ghost'] = -float(minGost) / weight

        scaredGhosts = self.getScaredGhost(gameState)
        scaredGhostPos = [scaredGhost.getPosition() for scaredGhost in scaredGhosts]
        if len(scaredGhostPos) > 0:
            minGost = min(self.getMazeDistance(myPosition, pos) for pos in scaredGhostPos)
            features['scaredGhost'] = -float(minGost) / weight


        # capsules features
        capsules = gameState.getCapsules()
        defendingCap = self.getCapsulesYouAreDefending(gameState)
        capsules = [cap for cap in capsules if cap not in defendingCap]

        if len(capsules) > 0:
            minDistanceOfCap = min(
                self.getMazeDistance(successor.getAgentPosition(self.index), capsule) for capsule in
                capsules)
            features['getCap'] = float(minDistanceOfCap) / weight

        # food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPosition, food) for food in foodList])
            features['distanceToFood'] = float(minDistance) / weight
        if gameState.isOnRedTeam(self.index):
            if len(gameState.getBlueFood().asList()) != 0:
                features['successorScore'] = -float(len(foodList)) / (len(gameState.getBlueFood().asList()) * 20)
        else:
            if len(gameState.getRedFood().asList()) != 0:
                features['successorScore'] = -float(len(foodList)) / (len(gameState.getRedFood().asList()) * 20)

        # back
        if (self.state == RUN and len(ghostPos) != 0 and minGost < 4) or (self.carryScore > 5) or gameState.data.timeleft < 60:
            if len(capsules) == 0 or (len(ghostPos) > 0 and minDistanceOfCap + 2 > minGost):
                features['getCap'] = 0
            features['distanceToFood'] = 0
            features['successorScore'] = 0
            features['ghost'] = 0
            distanceToBorder = self.getMinDistanceToMid(successor)
            features['back'] = -float(distanceToBorder) / weight

        return features


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = ()
        self.isTargetToFood = False

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        myFood = self.getFoodYouAreDefending(gameState).asList()

        if len(self.beforeFood) == 0:
            self.beforeFood == self.getFoodYouAreDefending(gameState).asList()

        features['onDefense'] = 1
        if myState.isPacman or gameState.getAgentState(self.index).scaredTimer > 0: features['onDefense'] = 0

        invaders = self.getInvaders(gameState)
        features['numInvaders'] = len(invaders)
        features['enemyDistance'] = 9999
        noEnemy = True
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
            noEnemy = False
            self.isTargetToFood = False

        if noEnemy:
            if len(self.beforeFood) > len(myFood):
                target = list(set(self.beforeFood) - set(myFood))[0]
                self.isTargetToFood = True
                self.target = target
                self.beforeFood = myFood
            else:
                borders = self.border
                minDistance = min((self.getMazeDistance(myPos, border) for border in borders))
                features['hasTarget'] = minDistance

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 0

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -50,
            'reverse': -2,
            'enemyDistance': -10,
            'hasTarget': -200
        }

    def aStarSearch(self, gameState, goal, start):
        """Search the node that has the lowest combined cost and heuristic first."""
        "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
        # test astarfoodsearch
        # init state
        priorityQueue = util.PriorityQueue()
        visited = []
        fourActions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        priorityQueue.push((start, [], 0), 0)

        legalPositions = [p for p in gameState.getWalls().asList(False)]

        # just calculate cost and push into priority queue
        while not priorityQueue.isEmpty():
            state, act, totalCost = priorityQueue.pop()
            if state == goal:
                return act
            else:
                successors = []
                for action in fourActions:
                    x, y = state
                    dx, dy = Actions.directionToVector(action)
                    nextx, nexty = int(x + dx), int(y + dy)
                    position = (nextx, nexty)
                    if position in legalPositions:
                        nextAction = action
                        successors.append((position, nextAction, 1))
                    else:
                        continue
                for successor, action, cost in successors:
                    # calculate cost
                    nextCost = totalCost + cost
                    heuristic = self.getMazeDistance(successor, goal)
                    if successor not in visited:
                        visited.append(successor)
                        nextAction = act + [action]
                        # get total cost * weight to plus heuristic cost
                        nextWeight = nextCost + heuristic
                        priorityQueue.push((successor, nextAction, nextCost), nextWeight)
        return None

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)
        currentPos = gameState.getAgentState(self.index).getPosition()
        invaders = self.getInvaders(gameState)

        actions.remove(Directions.STOP)

        # use astar to find the disapeared dot
        if self.isTargetToFood and len(invaders) == 0:
            actionList = self.aStarSearch(gameState, self.target, currentPos)

            if actionList is not None and len(actionList) > 0:
                return actionList[0]

        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)
