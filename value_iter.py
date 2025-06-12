#! python3

from gridworld import GridWorld

import numpy as np
from collections import defaultdict 

def value_iter_onestep(env, V, gamma) :
  for state in env.state :

    action_values = []
    for action in env.actions:
      next_state = env.next_state(state, action)
      r = env.reward(state, action, next_state)
      v = r + gamma * V[next_state]
      action_values.append(v)
    
    V[state] = max(action_values)
  return V

def value_iter(env, V, gamma, threshold=0.001) :
  for i in range(1000) :
    old_V = V.copy()
    V = value_iter_onestep(env, V, gamma)

    max_delta = None
    for state, value in V.items() :
      delta = abs(value - old_V[state])
      if max_delta is None or delta > max_delta :
        max_delta = delta
    
    if max_delta < threshold :
      break
  return V