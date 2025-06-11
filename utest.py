#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math

from gridworld import *
from policy_eval import *
from policy_iter import *

class FunctionTest(unittest.TestCase) :
  def test_property(self) :
    env = GridWorld()
    self.assertEqual(env.width, 4)
    self.assertEqual(env.high, 3)
    self.assertEqual(env.shape, (3, 4))

  def test_policy_eval(self) :
    # key是状态，value是此状态下策略
    pi = defaultdict(lambda : {0:0.25, 1:0.25, 2:0.25, 3:0.25})
    # key是状态，value是此状态下的价值
    V = defaultdict(lambda : 0)
    gamma = 0.9
    env = GridWorld()

    V = policy_evaluate(pi, V, env, gamma)
    env.render_v("test_policy_eval.png", V, pi)

  def test_policy_iter(self) :
    gamma = 0.9
    env = GridWorld()

    policy_iter(env, gamma, True)


unittest.main()