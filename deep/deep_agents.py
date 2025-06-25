#! python3

from deone import *
from qnet import QNet
from one_hot import one_hot
from collections import defaultdict

class QLearningNetAgent :
  def __init__(self) :
    self.gamma = 0.9
    self.action_size = 4
    self.action_space = [0, 1, 2, 3]

    self.alpha = 0.8
    self.epsilon = 0.1

    self.qnet = QNet()
    self.optimizer = SDG(lr=0.01)
    self.optimizer.setup(self.qnet)

  def to_q(self, env) :
    Q = defaultdict(lambda : 0)
    for state in env.state :
      action_value = self.qnet(one_hot(state))
      for action in self.action_space :
        Q[state, action] = action_value[0, action].data
    return Q

  def get_action(self, state) :
    if np.random.rand() < self.epsilon :
      action = np.random.choice(self.action_space)
    else :
      q = self.qnet(state)
      action = q.data.argmax()
    return action

  def update(self, state, action, reward, next_state, done) :
    if done :
      max_qs = np.zeros(1, dtype=np.float32)
    else :
      qs = self.qnet(one_hot(next_state))
      max_qs = qs.max(axis=1)
      max_qs.unchain()

    target = reward + max_qs * self.gamma
    action_value = self.qnet(one_hot(state))#.max(axis=1)
    # action是随机的也可能是最大的，总之是get_action时候得出的，这里就是要更新这个action的value
    x = action_value[:, action]
    loss = mean_square_error(x, target)

    self.qnet.cleargrad()
    loss.backward()
    self.optimizer.update()

    return loss.data
