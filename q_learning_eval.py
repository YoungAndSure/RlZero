
from collections import defaultdict 
import numpy as np
from gridworld import GridWorld
from agents import *

def q_learning_evaluate_q() :
  env = GridWorld()
  agent = QLearningAgent()

  for i in range(10000) :
    state = env.reset()

    while True :
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.update(state, action, reward, next_state, done)
      if done :
        break
      state = next_state
  env.render_q("test_q_learning_evaluate_q.png", agent.Q)