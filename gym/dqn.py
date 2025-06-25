#! python3

from deone import *
from replay_buffer import ReplayBuffer
import copy
import gym
import matplotlib.pyplot as plt
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class QNet(Model) :
  def __init__(self, action_size) :
    super().__init__()
    self.l1 = Linear(128)
    self.l2 = Linear(128)
    self.l3 = Linear(action_size)

  def forward(self, x) :
    return self.l3(relu(self.l2(relu(self.l1(x)))))

class DQNAgent :
  def __init__(self) :
    self.gamma = 0.9
    self.alpha = 0.8
    self.epsilon = 0.1

    self.action_size = 2

    self.buffer_size = 100000
    self.batch_size = 32
    self.rb = ReplayBuffer(self.batch_size, self.buffer_size)

    self.qnet = QNet(self.action_size)
    self.qnet_target = QNet(self.action_size)
    self.optimizer = SDG(lr=0.0005)
    self.optimizer.setup(self.qnet)

  def get_action(self, state) :
    if np.random.rand() < self.epsilon :
      return np.random.choice(self.action_size)
    else :
      state = state[np.newaxis, :]
      action_value = self.qnet(state)
      # NOTE:这不用设置axis?
      return action_value.data.argmax()

  def sync_qnet(self) :
    # 相当于每过一段时间就设置一个目标，后面往这个目标收敛，然后再找下一个目标
    self.qnet_target = copy.deepcopy(self.qnet)

  def update(self, state, action, reward, next_state, done) :
    self.rb.add(state, action, reward, next_state, done)
    if len(self.rb) < self.batch_size :
      return

    states, actions, rewards, next_states, dones = self.rb.get_batch()

    next_qs = self.qnet_target(next_states)
    next_qs.unchain()
    next_qs = next_qs.data.max(axis=1)
    targets = rewards + (1 - done) * self.gamma * next_qs

    qs = self.qnet(states)
    qs = qs[np.arange(self.batch_size), actions]
    loss = mean_square_error(qs, targets)

    self.qnet.cleargrad()
    loss.backward()
    self.optimizer.update()

sync_interval = 20
env = gym.make('CartPole-v0', render_mode='rgb_array')
agent = DQNAgent()
reward_history = []

for i in range(1000) :
  done = False
  state, info = env.reset()
  total_reward = 0

  while not done :
    action = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    agent.update(state, action, reward, next_state, done)
    state = next_state

    total_reward += reward
  reward_history.append(total_reward)

  if i % sync_interval == 0 :
    agent.sync_qnet()

plt.xlabel("")
plt.ylabel("reward")
plt.plot(reward_history)
plt.savefig('dqn.png')