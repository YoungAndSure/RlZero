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
bandit = Bandit(arms_num)
avg = 0
for i in range(arms_num) :
  arm = np.random.randint(10)
  reward = bandit.play(arm)
  avg = avg + (reward - avg) * (1 / (i + 1))
print(avg)