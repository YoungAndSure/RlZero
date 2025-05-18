#! python
import numpy as np

np.random.seed(0)
rewards = []
for i in range(11) :
  reward = np.random.rand()
  rewards.append(reward)
  avg = sum(rewards) / len(rewards)
  print(reward, avg)