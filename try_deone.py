#! python3

from deone import *

x_np = np.array(5.0)
x = Variable(x_np)
y = 3 * x ** 2
print(y)

y.backward()
print(x.grad)

# 向量的内积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(a), Variable(b) # 可以省略
c = matmul(a, b)
print(c)
# 矩阵的乘积
a =np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = matmul(a, b)
print(c)

def rosenbrock(x0, x1):
  y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
  return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 #学习率
iters = 10000 #重复次数
for i in range(iters):
  y = rosenbrock(x0, x1)
  x0.cleargrad()
  x1.cleargrad()
  y.backward()
  x0.data -= lr * x0.grad.data
  x1.data -= lr * x1.grad.data
print(x0, x1)


''' right answer:
variable(75.0)
variable(30.0)
variable(32)
variable([[19 22]
 [43 50]])
variable(-2.0) variable(400.0)
variable(0.9944984367782456) variable(0.9890050527419593)
'''