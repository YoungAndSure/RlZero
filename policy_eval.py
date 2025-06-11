#! python3

from gridworld import GridWorld

import numpy as np
from collections import defaultdict 

def eval_one_step(pi, V, env, gamma) :
  for state in env.state :
    if state == env.goal_state :
      V[state] = 0
      continue

    new_V = 0.0
    action_probs = pi[state]
    for action, action_prob in action_probs.items() :
      next_state = env.next_state(state, action)
      r = env.reward(state, action, next_state)
      new_V += action_prob * (r + gamma * V[next_state])
    V[state] = new_V

  return V

def policy_evaluate(pi, V, env, gamma, threshold=0.001) :
  for i in range(1000) :
    old_V = V.copy()
    V = eval_one_step(pi, V, env, gamma)

    max_delta = None
    for state in V.keys() :
      delta = abs(V[state] - old_V[state])
      if (max_delta is None or delta > max_delta) :
        max_delta = delta
    
    if max_delta < threshold :
      print("max step:{}".format(i))
      break
  return V