
from collections import defaultdict 
import numpy as np
from gridworld import GridWorld
from agents import *


def mc_evaluate() :
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

  return agent.V