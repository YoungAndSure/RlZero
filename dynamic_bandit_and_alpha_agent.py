#! python3
import numpy as np
import matplotlib.pyplot as plt

class DynamicBandit :
  def __init__(self, action_num, seed=0) :
    np.random.seed(seed)
    self.rates = np.random.rand(action_num)
    self.action_num = action_num

  def play(self, action) :
    self.rates += 0.1 * np.random.randn(self.action_num)
    if np.random.rand() > self.rates[action] :
    #if np.random.rand() > np.random.rand() :
      return 0
    return 1

  def get_rates(self) :
    return self.rates

class AlphaAgent :
  def __init__(self, epsilon, alpha, action_num=10) :
    self.epsilon = epsilon
    self.action_num = action_num
    self.ns = np.zeros((action_num))
    self.qs = np.zeros((action_num))
    self.alpha = alpha
  
  def update(self, action, reward) :
    self.ns[action] += 1
    if self.alpha is None :
      self.qs[action] = self.qs[action] + (reward - self.qs[action]) / (self.ns[action] + 1)
    else :
      self.qs[action] = self.qs[action] + (reward - self.qs[action]) * self.alpha

  def get_action(self) :
    if np.random.rand() < self.epsilon :
      return np.random.randint(self.action_num)
    return np.argmax(self.qs)

  def get_rewards(self) :
    return self.ns, self.qs

def play(play_times, action_num, epsilon, alpha, run_times, plt1, plt2) :
  all_rates = np.zeros((run_times, play_times))
  all_rewards = np.zeros((run_times, play_times))
  for run in range(run_times) :
    bandit = DynamicBandit(action_num, seed=run)
    agent = AlphaAgent(epsilon, alpha)

    total_reward = 0.0
    total_rewards = np.zeros(play_times)
    rates = np.zeros(play_times)
    for i in range(play_times) :
      action = agent.get_action()
      reward = bandit.play(action)
      agent.update(action, reward)

      total_reward += reward
      total_rewards[i] = total_reward
      rates[i] = (total_reward / (i + 1))
    all_rates[run] = rates
    all_rewards[run] = total_rewards
  avg_rates = np.average(all_rates, axis=0)
  avg_rewards = np.average(all_rewards, axis=0)

  plt1.set_xlabel('Steps')
  plt1.set_ylabel('TotalRewards')
  plt1.plot(avg_rewards, label='alpha:{}'.format(alpha))
  plt1.legend(loc='lower right')

  plt2.set_xlabel('Steps')
  plt2.set_ylabel('AvgRates')
  plt2.plot(avg_rates, label="alpha:{}".format(alpha))
  plt2.legend(loc='lower right')

fig, axes = plt.subplots(2, 1, figsize=(20, 20)) 
play(1000, 10, 0.1, None, 200, axes[0], axes[1])
play(1000, 10, 0.1, 0.8, 200, axes[0], axes[1])
plt.savefig("two")
