#! python3

import numpy as np

def one_hot(state) :
  HIGH, WIDTH = 3,4
  encode = np.zeros(HIGH * WIDTH, dtype=np.float32)
  idx = state[0] * WIDTH + state[1]
  encode[idx] = 1.0
  return encode[np.newaxis, :]

state = [2,0]
encode = one_hot(state)
print(encode)
print(encode.shape)