#! python3

from gridworld import GridWorld
from policy_eval import policy_evaluate

import numpy as np
from collections import defaultdict 

def argmax(kv) :
  max_arg = None
  max_value = None
  for (k, v) in kv.items() :
    if max_arg is None or max_value is None or v > max_value:
      max_arg = k
      max_value = v
  return max_arg

def greedy_policy(env, V, pi, gamma) :
  for state in env.state :

    action_value = {}
    for action in env.actions :
      next_state = env.next_state(state, action)
      r = env.reward(state, action, next_state)
      action_value[action] = r + gamma * V[next_state]
    max_action = argmax(action_value)
    action_prob = pi[state]
    action_prob[max_action] = 1.0

  return pi