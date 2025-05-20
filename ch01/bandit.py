#! python
import numpy as np

class Bandit :
  def __init__(self, arms) :
    np.random.seed(0)
    self.rates = np.random.rand(arms)

  def play(self, arm) :
    if np.random.rand() > self.rates[arm] :
      return 0
    return 1

arms_num = 10
ns = np.zeros((arms_num))
qs = np.zeros((arms_num))

play_times = 10
bandit = Bandit(arms_num)
for i in range(play_times) :
  arm = np.random.randint(arms_num)
  reward = bandit.play(arm)

  ns[arm] += 1
  qs[arm] = qs[arm] + (reward - qs[arm]) / (ns[arm] + 1)
print(ns)
print(qs)