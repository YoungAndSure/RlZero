#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math

from gridworld import *

class FunctionTest(unittest.TestCase) :
  def test_property(self) :
    env = GridWorld()
    self.assertEqual(env.width, 4)
    self.assertEqual(env.high, 3)
    self.assertEqual(env.shape, (3, 4))

unittest.main()