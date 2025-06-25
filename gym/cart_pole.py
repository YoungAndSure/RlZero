#! python3

import gym
import numpy as np
from PIL import Image
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

env = gym.make('CartPole-v0', render_mode="rgb_array")
state = env.reset()
print(state)

action_space = env.action_space
print(type(action_space))

done = False
frames = []
while not done :
  frame = env.render()
  img = Image.fromarray(frame)
  frames.append(img)
  action = np.random.choice([0, 1])
  next_state, reward, done, truncated, info = env.step(action)
env.close()

frames[0].save(
    "cartpole.gif", 
    save_all=True, 
    append_images=frames[1:], 
    duration=100,  # 每帧显示时间（毫秒）
    loop=0         # 0 表示无限循环
)