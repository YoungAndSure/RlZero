
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

def greedy_prob(Q, state, epsilon=0.0, action_space=[0,1,2,3]) :
  qv = {}
  for action in action_space :
    key = (state, action)
    qv[action] = Q[key]

  max_action = argmax(qv)

  base_prob = epsilon / len(action_space)
  # 开始有点疑惑，每次都重新搞个新的actions_prob，和固定值有啥区别？这是没理解actions_prob的含义
  # 这里设置的actions_prob，会影响下一次游戏选择action的概率
  # 如果设置为0和1，那一旦选到一条好路之后，后面每次都100%选这条路走，其他路就没机会走了，也就得不到其他非本条路的状态的价值
  # 所以设置的是“概率”，让好的路走的概率高，其他路走的概率低。
  # eplilon算法的含义就是调整各种选择的概率
  actions_prob = {}
  for action in action_space :
    actions_prob[action] = base_prob
  actions_prob[max_action] += 1 - epsilon

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
    self.memory = []

    self.epsilon = 0.1
    self.alpha = 0.1

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
      self.Q[key] += (G - self.Q[key]) * self.alpha

      self.pi[state] = greedy_prob(self.Q, state, self.epsilon, self.action_space)

class TdAgent :
  def __init__(self) :
    self.gamma = 0.9
    self.action_size = 4

    random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    self.pi = defaultdict(lambda : random_actions)
    self.V = defaultdict(lambda : 0)

    self.alpha = 0.1

  def get_action(self, state) :
    actions_prob = self.pi[state]
    actions = list(actions_prob.keys())
    probs = list(actions_prob.values())
    action = np.random.choice(actions, p=probs)
    return action

  def eval(self, state, reward, next_state, done) :
    next_V = 0 if done else self.V[next_state]
    G = reward + self.gamma * next_V
    self.V[state] += (G - self.V[state]) * self.alpha