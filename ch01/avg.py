#! python
import numpy as np

data_num = 3

np.random.seed(0)
rewards = []
for i in range(data_num) :
  reward = np.random.rand()
  rewards.append(reward)

avg = sum(rewards) / len(rewards)
print(avg)

avg2 = 0.0
for i in range(data_num) :
  update_rate = 1 / (i + 1)
  # NOTE: 
  avg2 += (rewards[i] - avg2) * update_rate
print(avg2)

assert(avg == avg2)