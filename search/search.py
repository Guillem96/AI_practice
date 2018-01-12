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
import node
import sys

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
    fringe = util.Stack()
    fringe.push(node.Node(problem.getStartState()))
    generated = {}
    while True:
        if fringe.isEmpty(): print 'no solution' ; sys.exit(-1)
        n = fringe.pop()
        generated[n.state] = n.state
        for s,a,c in problem.getSuccessors(n.state):
             ns = node.Node(s,n,a,c)
             if s not in generated:
                 if problem.isGoalState(ns.state): return ns.path()
                 fringe.push(ns)
                 generated[s] = []


def breadthFirstSearch(problem):
    fringe = util.Queue()
    fringe.push(node.Node(problem.getStartState()))
    generated = {}
    while True:
        if fringe.isEmpty(): print 'no solution' ; sys.exit(-1)
        n = fringe.pop()
        generated[n.state] = n.state
        for s,a,c in problem.getSuccessors(n.state):
             ns = node.Node(s,n,a,c)
             if s not in generated:
                 if problem.isGoalState(ns.state): return ns.path()
                 fringe.push(ns)
                 generated[s] = []

def uniformCostSearch(problem):
    def push(n):
        fringe.push(n,n.cost)
        generated[n.state] = [n,'F']
    fringe = util.PriorityQueue()
    generated = {}
    n = node.Node(problem.getStartState())
    push(n)
    while True:
        if fringe.isEmpty(): print 'no solution' ; sys.exit(-1)
        n = fringe.pop()
    	if problem.isGoalState(n.state): return n.path()
    	if generated[n.state][1] != 'E':
                generated[n.state] = [n, 'E']
                for s,a,c in problem.getSuccessors(n.state):
                    ns = node.Node(s,n,a,n.cost+c)
                    if ns.state not in generated:
                        push(ns)
                    elif generated[ns.state][1] == 'F' and ns.cost < generated[ns.state][0].cost:
                        push(ns)

def nullHeuristic(state, problem=None):
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    def push(n):
        fringe.push(n,n.cost)
        generated[n.state] = [n,'F']
    fringe = util.PriorityQueue()
    generated = {}
    n = node.Node(problem.getStartState())
    push(n)
    while True:
        if fringe.isEmpty(): print 'no solution' ; sys.exit(-1)
        n = fringe.pop()
        if problem.isGoalState(n.state): return n.path()
        if generated[n.state][1] != 'E': #Si no esta expandit
            generated[n.state] = [n, 'E']
            for s,a,c in problem.getSuccessors(n.state):
                fn = max(n.cost + c + heuristic(s, problem), n.cost + heuristic(n.state, problem)) #pathmax
                ns = Node(s,n,a,fn)
                if ns.state not in generated:
                    push(ns)
                elif ns.cost < generated[ns.state][0].cost: # guillem - >Saber perque no cal controlar que esta al fringe
                    push(ns)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
