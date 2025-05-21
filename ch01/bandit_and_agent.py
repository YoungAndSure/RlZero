#! python
import numpy as np

class Bandit :
  def __init__(self, action_num) :
    np.random.seed(0)
    self.rates = np.random.rand(action_num)

  def play(self, action) :
    if np.random.rand() > self.rates[action] :
      return 0
    return 1

  def get_rates(self) :
    return self.rates

class Agent :
  def __init__(self, epsilon, action_num=10) :
    self.epsilon = epsilon
    self.action_num = action_num
    self.ns = np.zeros((action_num))
    self.qs = np.zeros((action_num))
  
  def update(self, action, reward) :
    self.ns[action] += 1
    self.qs[action] = self.qs[action] + (reward - self.qs[action]) / (self.ns[action] + 1)

  def get_action(self) :
    if np.random.rand() < self.epsilon :
      return np.random.randint(self.action_num)
    return np.argmax(self.qs)

  def get_rewards(self) :
    return self.ns, self.qs


play_times = 10
# action_num意为有多少种action
action_num = 10
epsilon = 0.1
bandit = Bandit(action_num)
agent = Agent(epsilon)

for i in range(play_times) :
  action = agent.get_action()
  reward = bandit.play(action)
  agent.update(action, reward)
print(agent.get_rewards())
print(bandit.get_rates())