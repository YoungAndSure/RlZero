
from collections import defaultdict 
import numpy as np
from utils import argmax

class RandomAgent :
  def __init__(self) :
    self.gamma = 0.9
    self.action_size = 4

    random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    self.pi = defaultdict(lambda : random_actions)
    self.V = defaultdict(lambda : 0)
    self.cnts = defaultdict(lambda : 0)
    self.memory = []

  def get_action(self, state) :
    actions_prob = self.pi[state]
    actions = list(actions_prob.keys())
    probs = list(actions_prob.values())
    action = np.random.choice(actions, p=probs)
    return action

  def add(self, state, action, reward) :
    memory = (state, action, reward)
    self.memory.append(memory)

  def reset(self) :
    self.memory.clear()

  def eval(self) :
    G = 0
    for memory in reversed(self.memory) :
      state, action, reward = memory
      G = self.gamma * G + reward
      self.cnts[state] += 1
      self.V[state] += (G - self.V[state]) / self.cnts[state]

def greedy_prob(Q, state, action_space) :
  qv = {}
  for action in action_space :
    key = (state, action)
    qv[action] = Q[key]

  max_action = argmax(qv)

  actions_prob = {}
  for action in action_space :
    actions_prob[action] = 0
  actions_prob[max_action] = 1

  return actions_prob

class McAgent() :
  def __init__(self) :
    self.gamma = 0.9
    self.action_space = [0, 1, 2, 3]

    random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    self.pi = defaultdict(lambda : random_actions)
    # Q 与 V不同，V的key是state, value是状态价值
    # Q的key是(state, action)，value是行动价值，或者说“此状态下的行动价值”
    self.Q = defaultdict(lambda : 0)
    self.cnts = defaultdict(lambda : 0)
    self.memory = []

  def get_action(self, state) :
    actions_prob = self.pi[state]
    actions = list(actions_prob.keys())
    probs = list(actions_prob.values())
    action = np.random.choice(actions, p=probs)
    return action

  def add(self, state, action, reward) :
    memory = (state, action, reward)
    self.memory.append(memory)

  def reset(self) :
    self.memory.clear()

  def update(self) :
    G = 0
    for memory in reversed(self.memory) :
      state, action, reward = memory
      G = self.gamma * G + reward
      key = (state, action)
      self.cnts[key] += 1
      self.Q[key] += (G - self.Q[key]) / self.cnts[key]

      self.pi[state] = greedy_prob(self.Q, state, self.action_space)