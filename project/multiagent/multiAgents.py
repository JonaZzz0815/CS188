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

from gettext import lngettext
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # compute nearest food
        min_food_dis = 1e9
        food_left = len(newFood.asList())
        for food in newFood.asList():
            min_food_dis = min(min_food_dis,
                                manhattanDistance(newPos,food))
        if not newFood.asList():
            min_food_dis = 0
        # compute nearest ghost with newScaredTimes
        min_ghost_dis = 1e9
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                min_ghost_dis = min(min_ghost_dis,
                                manhattanDistance(newPos,ghost.getPosition()))
        penalty =11/(min_ghost_dis+1) + min_food_dis/5 + 3/(food_left+1)
        return  successorGameState.getScore() - penalty

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        return self.MinMaxSearch(gameState,agentIndex=0, depth=self.depth)[1]

    def greater(self,a,b):
        return a>b
    def less(self,a,b):
        return a<b
    def MinOrMax(self,gameState: GameState,agentIndex,depth,flag = "Min"):
        actions = gameState.getLegalActions(agentIndex)
        # init next agent and depth by agentIndex
        if agentIndex == gameState.getNumAgents() - 1 :
            nxt_agent,nxt_depth = 0 , depth - 1
        else:
            nxt_agent,nxt_depth = agentIndex+1 , depth
        # set init val by Min or Max
        if flag == "Min":
            aimed_score = 1e9
            comp = self.less
        else:
            aimed_score = -1e9
            comp = self.greater
        
        aimed_action = Directions.STOP

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex,action)
            now_score = self.MinMaxSearch(successor,nxt_agent,nxt_depth)[0]
            if comp(now_score,aimed_score):
                aimed_score,aimed_action = now_score,action

        return aimed_score,aimed_action

    def MinMaxSearch(self,gameState: GameState,agentIndex,depth):
        # base case:
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:# our player
            return self.MinOrMax(gameState,agentIndex,depth,"Max")
        else:
            return self.MinOrMax(gameState,agentIndex,depth,"Min")

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.AlphaBetaSearch(gameState,0, self.depth, -1e9, 1e9)[1]

    def AlphaBetaSearch(self,gameState, agentIndex, depth, a, b):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:# our player
            return self.AlphaValue(gameState, agentIndex, depth, a, b)
        else:
            return self.BetaValue(gameState, agentIndex, depth, a, b)
    
    def AlphaValue(self,gameState, agentIndex, depth, a, b):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1 :
            nxt_agent,nxt_depth = 0 , depth - 1
        else:
            nxt_agent,nxt_depth = agentIndex+1 , depth
        max_score = -1e9
        max_action = Directions.STOP
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex,action)
            now_score = self.AlphaBetaSearch(successor,nxt_agent,nxt_depth,
                                                                        a,b)[0]
            if now_score > max_score:
                max_score =  now_score
                max_action = action
            if now_score > b:
                return now_score,action
            a = max(a,max_score)
        return max_score,max_action
    
    def BetaValue(self,gameState, agentIndex, depth, a, b):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1 :
            nxt_agent,nxt_depth = 0 , depth - 1
        else:
            nxt_agent,nxt_depth = agentIndex+1 ,depth
        min_score = 1e9
        min_action = Directions.STOP
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex,action)
            now_score = self.AlphaBetaSearch(successor,nxt_agent,nxt_depth,
                                                                        a,b)[0]
            if now_score < min_score:
                min_score =  now_score
                min_action = action
            if now_score < a:
                return now_score,action
            b = min(b,min_score)
        return min_score,min_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.ExpectiMaxSearch(gameState,0,self.depth)[1]
        
    def ExpectiMaxSearch(self,gameState: GameState,agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:# our player
            return self.MinOrMax(gameState, agentIndex, depth,"Max")
        else:
            return self.Expection(gameState, agentIndex, depth)
    

    def greater(self,a,b):
        return a>b
    def less(self,a,b):
        return a<b
    # we only use max here
    def MinOrMax(self,gameState: GameState,agentIndex,depth,flag = "Min"):
        actions = gameState.getLegalActions(agentIndex)
        # init next agent and depth by agentIndex
        if agentIndex == gameState.getNumAgents() - 1 :
            nxt_agent,nxt_depth = 0 , depth - 1
        else:
            nxt_agent,nxt_depth = agentIndex+1 , depth
        # set init val by Min or Max
        if flag == "Min":
            aimed_score = 1e9
            comp = self.less
        else:
            aimed_score = -1e9
            comp = self.greater
        
        aimed_action = Directions.STOP

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex,action)
            now_score = self.ExpectiMaxSearch(successor,nxt_agent,nxt_depth)[0]
            if comp(now_score,aimed_score):
                aimed_score,aimed_action = now_score,action

        return aimed_score,aimed_action

    def Expection(self,gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1 :
            nxt_agent,nxt_depth = 0 , depth - 1
        else:
            nxt_agent,nxt_depth = agentIndex+1 ,depth
        expected_score = 0
        expected_action = Directions.STOP
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex,action)
            now_score = self.ExpectiMaxSearch(successor,nxt_agent,nxt_depth)[0]
            expected_score += now_score
        expected_score /= len(actions)                                                          
            
        return expected_score,expected_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    Three Criterion:
    Nearest Food in manhattan distance: less is better
    Sum of unscared ghost's distance: more is better
    Sum of unscared ghost's distance: less is more
    
    the weight is :13,-5,3
    """
    "*** YOUR CODE HERE ***"
    
    Pos = currentGameState.getPacmanPosition()
    Foods = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    Capsules =  currentGameState.getCapsules()
    # compute nearest food
    nearest_food = 0
    food_dis = [manhattanDistance(Pos,food) for food in Foods.asList()]
    if len(food_dis) > 0:
        nearest_food = min(food_dis)
    # compute nearest ghost with newScaredTimes
    ghost_dis = []
    scared_dis = []
    for ghost in GhostStates:
        dis = manhattanDistance(Pos,ghost.getPosition())
        if ghost.scaredTimer > 0:
            ghost_dis.append(dis)
        else:
             scared_dis.append(dis)
    score = 13.0/(nearest_food+0.5)
    ghost_dis = sorted(ghost_dis)
    scared_dis = sorted(scared_dis)
    length1 = len(ghost_dis)
    length2 = len(scared_dis)
    score -= 5.0*sum([ (length1-i)/(ghost_dis[i]+0.5) for i in range(length1) ])
    score += 3.0*sum([ (length2-i)/(scared_dis[i]+0.5) for i in range(length2)])
    return currentGameState.getScore() + score

# Abbreviation
better = betterEvaluationFunction
