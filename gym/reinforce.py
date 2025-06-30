#! python3

import deone as D
import numpy as np
from replay_buffer import ReplayBuffer
import gym
import matplotlib.pyplot as plt
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class Policy(D.Model) :
    def __init__(self, ndim=128, action_size=2) :
      super().__init__()
      self.l1 = D.Linear(ndim)
      self.l2 = D.Linear(action_size)

    def forward(self, x) :
      y1 = D.relu(self.l1(x))
      y2 = D.softmax(self.l2(y1))
      return y2

class Agent() :
  def __init__(self) :
    self.gamma = 0.98
    self.lr = 0.0002
    self.action_size = 2
    self.memory = []

    self.pi = Policy()
    self.optimizer = D.SDG(self.lr)
    self.optimizer.setup(self.pi)

  def get_action(self, state) :
    state = state[np.newaxis, :]
    action_probs = self.pi(state)
    actions = np.arange(self.action_size)
    probs = action_probs[0, actions]
    action = np.random.choice(actions, p=probs.data)
    return action, probs[action]

  def add(self, reward, prob) :
    self.memory.append((reward, prob))

  def update(self) :
    G = 0
    loss = 0
    for reward, prob in reversed(self.memory) :
      G = reward + self.gamma * G
      loss += - G * D.log(prob)

    self.pi.cleargrad()
    loss.backward()
    self.optimizer.update()
    self.memory = []

agent = Agent()
env = gym.make('CartPole-v0', render_mode='rgb_array')
reward_history = []

for i in range(10000) :
  done = False
  state, info = env.reset()
  total_reward = 0

  while not done :
    action, prob = agent.get_action(state)
    agent.add(action, prob)
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    total_reward += reward
  agent.update()
  reward_history.append(total_reward)
avg_reward_history = []
for i in range(100) :
  avg = sum(reward_history[i * 100 : (i + 1) * 100]) / 100
  avg_reward_history.append(avg)

plt.xlabel("")
plt.ylabel("reward")
plt.plot(avg_reward_history)
plt.savefig('reinforce.png')