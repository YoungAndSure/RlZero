#! python3

import gym
import numpy as np
from PIL import Image
from replay_buffer import ReplayBuffer
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

rb = ReplayBuffer(batch_size=32, buffer_size=10000)

env = gym.make('CartPole-v0', render_mode="rgb_array")

for i in range(100) :
  state, info = env.reset()
  done = False
  frames = []

  while not done :
    frame = env.render()
    img = Image.fromarray(frame)
    frames.append(img)

    action = np.random.choice([0, 1])
    next_state, reward, done, truncated, info = env.step(action)
    rb.add(state, action, reward, next_state, done)
    state = next_state
  env.close()

print("len:{}".format(len(rb)))

states, actions, rewards, next_states, dones = rb.get_batch()
print(states.shape)
print(actions.shape)
print(rewards.shape)
print(next_states.shape)
print(dones.shape)

frames[0].save(
    "cartpole.gif", 
    save_all=True, 
    append_images=frames[1:], 
    duration=100,  # 每帧显示时间（毫秒）
    loop=0         # 0 表示无限循环
)