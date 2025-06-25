#! python3

if '__file__' in globals() :
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gridworld import GridWorld
from deep_agents import QLearningNetAgent
from one_hot import one_hot

agent = QLearningNetAgent()
env = GridWorld()
loss_history = []

for i in range(1000) :
  state = env.reset()
  total_loss = 0.0
  cnt = 0

  while True :
    action = agent.get_action(one_hot(state))
    next_state, reward, done = env.step(action)
    loss = agent.update(state, action, reward, next_state, done)
    total_loss += loss
    cnt += 1

    # 为什么这里直接就break了，之前有一个还要再处理一下？
    # 之前那个是存下来，当得到了next_state的时候计算state，
    # 现在是在state时候用next_state计算state，所以当done的时候，就计算完了
    if done :
      break

    state = next_state
  loss_history.append(total_loss / cnt)

Q = agent.to_q(env)
env.render_q("q_learning_net.png", Q)