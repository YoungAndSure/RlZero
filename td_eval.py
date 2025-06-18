
from collections import defaultdict 
import numpy as np
from gridworld import GridWorld
from agents import *


def td_evaluate_v() :
  env = GridWorld()
  agent = TdAgent()

  for i in range(1000) :
    state = env.reset()

    while True :
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.eval(state, reward, next_state, done)
      if done :
        break
      state = next_state
  env.render_v("test_td_evaluate.png", agent.V)