#! python3

from deone import *
from one_hot import one_hot

class QNet(Model) :
  def __init__(self) :
    super().__init__()
    self.l1 = Linear(100)
    self.l2 = Linear(4)

  def forward(self, x) :
    y1 = self.l1(x)
    y2 = relu(y1)
    y3 = self.l2(y2)
    return y3

model = QNet()
state = (2, 0)
encode = one_hot(state)
y = model(encode)
print(y)