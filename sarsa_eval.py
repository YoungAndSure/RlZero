
from collections import defaultdict 
import numpy as np
from gridworld import GridWorld
from agents import *

def sarsa_evaluate_q() :
  env = GridWorld()
  agent = SarsaAgent()

  for i in range(10000) :
    state = env.reset()
    agent.reset()

    while True :
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.update(state, action, reward, done)
      if done :
        agent.update(state, None, None, None)
        break
      state = next_state
  env.render_q("test_sarsa_evaluate_q.png", agent.Q)

def sarsa_offpolicy_evaluate_q() :
  env = GridWorld()
  agent = SarsaOffPolicyAgent()

  for i in range(10000) :
    state = env.reset()
    agent.reset()

    while True :
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.update(state, action, reward, done)
      if done :
        agent.update(state, None, None, None)
        break
      state = next_state
  env.render_q("test_sarsa_offpolicy_evaluate_q.png", agent.Q)