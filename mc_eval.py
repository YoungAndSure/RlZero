
from collections import defaultdict 
import numpy as np
from gridworld import GridWorld
from agents import *


def mc_evaluate_v() :
  env = GridWorld()
  agent = RandomAgent()

  for i in range(1000) :
    state = env.reset()
    agent.reset()

    while True :
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.add(state, action, reward)
      if done :
        agent.eval()
        break
      state = next_state
  env.render_v("test_mc_evaluate.png", agent.V)

def mc_evaluate_q() :
  env = GridWorld()
  agent = McAgent()

  for i in range(10000) :
    state = env.reset()
    agent.reset()

    while True :
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.add(state, action, reward)
      if done :
        agent.update()
        break
      state = next_state
  env.render_q("test_mc_evaluate_q.png", agent.Q, agent.pi)