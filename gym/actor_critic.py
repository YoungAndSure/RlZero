#! python3

import deone as D
import numpy as np
from replay_buffer import ReplayBuffer
import gym
import matplotlib.pyplot as plt
from collections import deque
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class PolicyNet(D.Model) :
  def __init__(self, ndim=128, action_size=2) :
    super().__init__()
    self.l1 = D.Linear(ndim)
    self.l2 = D.Linear(action_size)

  def forward(self, x) :
    y1 = D.relu(self.l1(x))
    y2 = D.softmax(self.l2(y1))
    return y2

class ValueNet(D.Model) :
  def __init__(self) :
    super().__init__()
    self.l1 = D.Linear(128)
    self.l2 = D.Linear(1)

  def forward(self, x) :
    y1 = D.relu(self.l1(x))
    y2 = self.l2(y1)
    return y2

class Agent() :
  def __init__(self) :
    self.gamma = 0.98
    self.lr = 0.0002
    self.action_size = 2

    self.pi = PolicyNet()
    self.pi_optimizer = D.SDG(self.lr)
    self.pi_optimizer.setup(self.pi)

    self.V = ValueNet()
    self.V_optimizer = D.SDG(self.lr)
    self.V_optimizer.setup(self.V)

  def get_action(self, state) :
    state = state[np.newaxis, :]
    action_probs = self.pi(state)
    action = np.random.choice(len(action_probs[0]), p=action_probs[0].data)
    # 保持多batch的shape，不然后面loss_pi反向传播会失败
    return action, action_probs[:, action]

  def update(self, state, reward, prob, next_state, done) :
    state = state[np.newaxis, :]
    next_state = next_state[np.newaxis, :]

    target = reward + self.gamma * self.V(next_state) * (1 - done)
    target.unchain()
    v = self.V(state)
    loss_v = D.mean_square_error(v, target)

    delta = target - v
    delta.unchain()
    loss_pi = - D.log(prob) * delta

    self.V.cleargrad()
    loss_v.backward()
    self.V_optimizer.update()

    self.pi.cleargrad()
    loss_pi.backward()
    self.pi_optimizer.update()

agent = Agent()
env = gym.make('CartPole-v0', render_mode='rgb_array')
reward_history = []

for i in range(10000) :
  done = False
  state, info = env.reset()
  total_reward = 0

  while not done :
    action, prob = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    agent.update(state, reward, prob, next_state, done)
    state = next_state
    total_reward += reward
  reward_history.append(total_reward)
avg_reward_history = []
for i in range(100) :
  avg = sum(reward_history[i * 100 : (i + 1) * 100]) / 100
  avg_reward_history.append(avg)

plt.xlabel("")
plt.ylabel("reward")
plt.plot(avg_reward_history)
plt.savefig('actor_critic.png')