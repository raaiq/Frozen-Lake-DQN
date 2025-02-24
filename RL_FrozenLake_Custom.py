import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

#TODO: Measure how many steps to find path for 4x4 and 8x8. 
#TODO: Try to modify implementation with NN to use negative feedback. Terminated is true with reward as well. I didn't know that.


#TODO: Refactor to make code cleaner, e.g there's some code duplication and last minute logic. That can be clearned up and more organized
#TODO: There's a bug where the player resets when it encounters a loop, even if the loop doesn't lead it to the beginning. Instead of resetting it should plan a new path from it's current state



possibleActions = [0,1,2,3]
# Maybe have seperate object for workspace variables
knownStates = {}

class State():
    def __init__(self, number):
        self.actionMap = {}
        self.statesToAvoid = set()
        self.is_terminal = False
        self.id  = number

#Might just need the states not the action
    def availableActionState(self):
        return {a:s for (a,s) in self.actionMap.items() if s.id != self.id and s not in self.statesToAvoid}
    
    def unexploredActions(self):
        return [a for  a in possibleActions if a not in self.actionMap.keys()]
    
    def getActionForState(self, state):
        for a,s in self.actionMap.items():
            if (s.id == state.id):
                return a
        return None
    def avoidState(self, state):
        self.statesToAvoid.add(state)

    def shouldAvoid(self):
        return self.is_terminal or len(self.statesToAvoid) == 4
    
    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return self.id == other.id

def complementAction(action):
    return (action+2) % 4

# Returns he first valid path it finds.
# First state in path isn't included
# This method can be very memory intensive and inefficent
def exploreStateRecursive(stateNo, pathToDest:list):
    state = knownStates[stateNo]
    if state is None:
        raise Exception("No valid state found")
    if (pathToDest.count(stateNo) > 0):
        return ()
    pathToDest.append(stateNo)
    unexploredActions = state.unexploredActions()
    if (len(unexploredActions) > 0):
        return (unexploredActions[0], pathToDest)
    else: 
        for _, nState in state.availableActionState().items():
            possiblePath = copy.deepcopy(pathToDest)
            result = exploreStateRecursive(nState.id, possiblePath)
            if (len(result) >0):
                return result
    return ()

        
   # Right now this just returns a random path, not the path most exploited. 
   # In future make this function return path most exploited to know whether it's a valid path or not. 
   # If not then use any valueble insights from that path to construct new possible path
def exploreState(state:State):
    pathToDest = list()
    return exploreStateRecursive(state.id, pathToDest)

def eliminateBadState(stateOfInterest:State):
    actionStatePair=stateOfInterest.availableActionState()
    for _, state in actionStatePair.items():
        state.avoidState(stateOfInterest)
        if(state.shouldAvoid()):
            eliminateBadState(state)


if __name__ == '__main__':
    env= gym.make('FrozenLake-v1',map_name="4x4",is_slippery=False,render_mode= "human")
    firstState = env.reset()[0]
    currentState = State(firstState)
    knownStates[firstState] = currentState
    total_steps = 0 
    while True:
        targetAction, pathToTarget = exploreState(currentState)

        if (targetAction is None):
            raise Exception("This isn't possible: no state found to be explored")
        
        # There's common code in traversing exploring path and traversing unexplored path. Maybe can combine them
        for i in range(1,len(pathToTarget)):
            s = knownStates[pathToTarget[i]]
            action = currentState.getActionForState(s)
            if action is None:
                raise Exception("""How this possibe ?! exploreState said this was a valid lead-out state, 
                                but current state couldn't find the action for it""")
            newStateNo, _,_,_,_ = env.step(action)
            total_steps += 1
            currentState = knownStates.get(newStateNo)
            if (newStateNo is not s.id):
                raise Exception("Not on the predetermined path. What went wrong?!?!")
        path = pathToTarget

        while True:
            newStateNo, reward, terminated, truncated, _ = env.step(targetAction)
            total_steps += 1
            newState:State = knownStates.get(newStateNo,State(newStateNo))
            knownStates[newStateNo] = newState
            currentState.actionMap[targetAction] = newState
            if currentState.id is not newStateNo:
                newState.actionMap[complementAction(targetAction)] = currentState
            
            if terminated:
                if reward != 0:
                    print(f'This is path to reward:{path}')
                    print(f"It took {total_steps} steps to get the reward")
                    sys.exit()
                else:
                    newState.is_terminal = True
                    eliminateBadState(newState)
                    # Below logic can be made cleaner
                    path = []
                    firstState = env.reset()[0]
                    currentState = knownStates.get(firstState)
                    if currentState is None:
                        raise Exception("This ain't possible ?! First state not found in known state map")
                    break
            if currentState.id != newState.id:
                if path.count(newState.id):
                    path = []
                    break
                else:
                    path.append(newState.id)
            if newStateNo >= 12:
                t=1
            
            if truncated:
                raise Exception("There's supposed to be no truncation")
            
            currentState = newState
            a =currentState.unexploredActions()
            if (len(a) == 0):
                path = []
                firstState = env.reset()[0]
                currentState = knownStates.get(firstState)
                if currentState is None:
                    raise Exception("This ain't possible ?! First state not found in known state map")
                break
            targetAction = a[0]
    