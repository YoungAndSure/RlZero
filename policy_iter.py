#! python3

from gridworld import GridWorld
from policy_eval import policy_evaluate

import numpy as np
from collections import defaultdict 
from utils import argmax

# pi是通过当前的V找到的最优策略，和前边的V前边的pi都没有关系，所以这里不要传入之前的pi
def greedy_policy(env, V, gamma) :
  new_pi = {}
  for state in env.state :
    action_value = {}
    for action in env.actions :
      next_state = env.next_state(state, action)
      r = env.reward(state, action, next_state)
      action_value[action] = r + gamma * V[next_state]
    max_action = argmax(action_value)
    #action_prob = new_pi[state]
    action_prob = {0:0.0,1:0.0,2:0.0,3:0.0}
    action_prob[max_action] = 1.0
    new_pi[state] = action_prob

  return new_pi

# 策略迭代法
def policy_iter(env, gamma, is_render=False) :
  gamma = 0.9
  env = GridWorld()
  V = defaultdict(lambda : 0.0)
  pi = defaultdict(lambda : {0:0.25, 1:0.25, 2:0.25, 3:0.25})

  for i in range(1000) :
    V = policy_evaluate(pi, V, env, gamma)
    new_pi = greedy_policy(env, V, gamma)

    if is_render :
      env.render_v("test_policy_iter.png", V, pi)

    if pi == new_pi :
      break

    pi = new_pi

  return pi