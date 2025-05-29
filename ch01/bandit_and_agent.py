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


play_times = 1000
# action_num意为有多少种action
action_num = 10
epsilon = 0.1
run_times = 200
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

plt.xlabel('Steps')
plt.ylabel('TotalRewards')
plt.plot(total_rewards)
plt.savefig("play1000_rewards")

plt.clf()
plt.xlabel('Steps')
plt.ylabel('Rates')
plt.plot(rates)
plt.savefig("play1000_rates")

plt.clf()
plt.xlabel('Steps')
plt.ylabel('AvgRates')
plt.plot(avg_rates)
plt.savefig("run200play1000_avg_rates")