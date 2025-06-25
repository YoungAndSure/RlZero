#! python3

from collections import deque
import numpy as np
import random

class ReplayBuffer :
  def __init__(self, batch_size, buffer_size=1000) :
    self.buffer_size = buffer_size
    self.buffer = deque(maxlen=self.buffer_size)
    self.batch_size = batch_size

  def add(self, state, action, reward, next_state, done) :
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self) :
    return len(self.buffer)

  def get_batch(self) :
    sample = random.sample(self.buffer, self.batch_size)

    state = np.stack([x[0] for x in sample])
    action = np.array([x[1] for x in sample])
    reward = np.array([x[2] for x in sample])
    next_state = np.stack([x[3] for x in sample])
    done = np.array([x[4] for x in sample]).astype(np.int32)

    return state, action, reward, next_state, done