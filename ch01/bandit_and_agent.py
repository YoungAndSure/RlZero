#! python3
import numpy as np
import matplotlib.pyplot as plt

class Bandit :
  def __init__(self, action_num, seed=0) :
    np.random.seed(seed)
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

def play(play_times, action_num, epsilon, run_times, plt1, plt2, plt3) :
  all_rates = np.zeros((run_times, play_times))
  for run in range(run_times) :
    bandit = Bandit(action_num, seed=run)
    agent = Agent(epsilon)

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
  avg_rates = np.average(all_rates, axis=0)

  plt1.set_title('epsilon:{}'.format(epsilon))
  plt1.set_xlabel('Steps')
  plt1.set_ylabel('TotalRewards')
  plt1.plot(total_rewards)

  plt2.set_title('epsilon:{}'.format(epsilon))
  plt2.set_xlabel('Steps')
  plt2.set_ylabel('Rates')
  plt2.plot(rates)

  plt3.set_xlabel('Steps')
  plt3.set_ylabel('AvgRates')
  plt3.plot(avg_rates, label="epsilon:{}".format(epsilon))
  plt3.legend(loc='lower right')

fig, axes = plt.subplots(4, 2, figsize=(20, 20)) 
play(1000, 10, 0.1, 200, axes[0, 0], axes[0, 1], axes[3,0])
play(1000, 10, 0.3, 200, axes[1, 0], axes[1, 1], axes[3, 0])
play(1000, 10, 0.01, 200, axes[2, 0], axes[2, 1], axes[3, 0])
plt.savefig("one")
