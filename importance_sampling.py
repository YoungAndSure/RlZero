#! python3

import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

exp = 0.0
for i in range(len(x)) :
  exp += x[i] * pi[i]
print(exp)
print()

n = 100
samples = []

# mc
for i in range(n) :
  s = np.random.choice(x, p=pi)
  samples.append(s)
exp_mc = np.mean(samples)
var_mc = np.var(samples)
print(exp_mc)
print(var_mc)
print()

samples = []
b = np.array([1/3.0, 1/3.0, 1/3.0])
for _ in range(n) :
  # 既要取值，又要取概率算权重，所以要对索引号采样
  idxs = np.arange(len(b))
  idx = np.random.choice(idxs, p=b)
  samples.append(s * (pi[idx] / b[idx]))
exp_is = np.mean(samples)
var_is = np.var(samples)
print(exp_is)
print(var_is)