
from collections import defaultdict 
from collections import deque
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

class SarsaAgent :
  def __init__(self) :
    self.gamma = 0.9
    self.action_size = 4
    self.action_space = [0, 1, 2, 3]

    random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    # 每个state都有一个！！！！
    self.pi = defaultdict(lambda : random_actions)
    self.Q = defaultdict(lambda : 0)

    self.alpha = 0.8
    self.epsilon = 0.1
    self.memory = deque(maxlen=2)

  def get_action(self, state) :
    actions_prob = self.pi[state]
    actions = list(actions_prob.keys())
    probs = list(actions_prob.values())
    action = np.random.choice(actions, p=probs)
    return action

  def reset(self) :
    self.memory.clear()

  def update(self, state, action, reward, done) :
    self.memory.append((state, action, reward, done))
    if len(self.memory) < 2 :
      return
    
    state, action, reward, done = self.memory[0]
    next_state, next_action, _, _ = self.memory[1]

    key = (state, action)
    next_q = 0 if done else self.Q[next_state, next_action]
    target = reward + next_q * self.gamma
    self.Q[key] += (target - self.Q[key]) * self.alpha

    self.pi[state] = greedy_prob(self.Q, state, self.epsilon, self.action_space)

class SarsaOffPolicyAgent :
  def __init__(self) :
    self.gamma = 0.9
    self.action_size = 4
    self.action_space = [0, 1, 2, 3]

    random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    self.pi = defaultdict(lambda : random_actions)
    self.b = defaultdict(lambda : random_actions)
    self.Q = defaultdict(lambda : 0)

    self.alpha = 0.8
    self.epsilon = 0.1
    self.memory = deque(maxlen=2)

  def get_action(self, state) :
    actions_prob = self.b[state]
    actions = list(actions_prob.keys())
    probs = list(actions_prob.values())
    action = np.random.choice(actions, p=probs)
    return action

  def reset(self) :
    self.memory.clear()

  def update(self, state, action, reward, done) :
    self.memory.append((state, action, reward, done))
    if len(self.memory) < 2 :
      return
    
    state, action, reward, done = self.memory[0]
    next_state, next_action, _, _ = self.memory[1]

    next_q = 0 if done else self.Q[next_state, next_action]
    rho = 1 if done else self.pi[next_state][next_action] / self.b[next_state][next_action]

    key = (state, action)
    target = (reward + next_q * self.gamma) * rho
    self.Q[key] += (target - self.Q[key]) * self.alpha

    # 行动策略用epsilon策略更新，目标策略用贪婪策略更新
    # 目标策略是最优策略，agent在过程中不会用到的，agent只会用到行动策略
    # 强化学习最终是生成了这个目标策略
    self.pi[state] = greedy_prob(self.Q, state, 0, self.action_space)
    self.b[state] = greedy_prob(self.Q, state, self.epsilon, self.action_space)

class QLearningAgent :
  def __init__(self) :
    self.gamma = 0.9
    self.action_size = 4
    self.action_space = [0, 1, 2, 3]

    random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    self.pi = defaultdict(lambda : random_actions)
    self.b = defaultdict(lambda : random_actions)
    self.Q = defaultdict(lambda : 0)

    self.alpha = 0.8
    self.epsilon = 0.1

  def get_action(self, state) :
    actions_prob = self.b[state]
    actions = list(actions_prob.keys())
    probs = list(actions_prob.values())
    action = np.random.choice(actions, p=probs)
    return action

  def update(self, state, action, reward, next_state, done) :

    action_value = []
    for action in self.action_space :
      action_value.append(self.Q[next_state, action])
    max_value = max(action_value)

    next_q = 0 if done else max_value

    key = (state, action)
    target = reward + next_q * self.gamma
    self.Q[key] += (target - self.Q[key]) * self.alpha

    self.pi[state] = greedy_prob(self.Q, state, 0, self.action_space)
    self.b[state] = greedy_prob(self.Q, state, self.epsilon, self.action_space)