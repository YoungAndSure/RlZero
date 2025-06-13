
import numpy as np
from gridworld_render import Renderer

class GridWorld :
  def __init__(self) :
    self.action_space = [0,1,2,3]
    self.action_meaning = ['UP','DOWN','LEFT','RIGHT']
    # fix, UP DOWN写反了，排查半天
    self.action_move = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    self.reward_map = np.array(
      [[0,0,0,1.0],
      [0,None,0,-1],
      [0,0,0,0]], dtype=np.float32
    )
    self.wall_state = (1,1)
    self.goal_state = (0,3)
    self.start_state = (2,0)
    self.agent_state = self.start_state

  @property
  def width(self) :
    return self.reward_map.shape[1]

  @property
  def high(self) :
    return self.reward_map.shape[0]

  @property
  def shape(self) :
    return self.reward_map.shape

  @property
  def actions(self) :
    return self.action_space

  @property
  def state(self) :
    # 计算期望的时候，需要遍历state，所以这里是for
    for h in range(self.high) :
      for w in range(self.width) :
        yield (h, w)

  def next_state(self, state, action) :
    move = self.action_move[action]
    next_state = (state[0] + move[0], state[1] + move[1])
    ns1, ns2 = next_state[0], next_state[1]

    if ns1 < 0 or ns1 >= self.high or ns2 < 0 or ns2 >= self.width or next_state == self.wall_state:
      next_state = state

    return next_state

  def reward(self, state, action, next_state) :
    return self.reward_map[next_state]

  def render_v(self, file_name, v=None, policy=None, print_value=True):
      renderer = Renderer(self.reward_map, self.goal_state,
                                        self.wall_state)
      renderer.render_v(file_name, v, policy, print_value)

  def render_q(self, file_name, q=None, print_value=True):
      renderer = Renderer(self.reward_map, self.goal_state,
                                        self.wall_state)
      renderer.render_q(file_name, q, print_value)

  def reset(self) :
    self.agent_state = self.start_state
    return self.agent_state

  def step(self, action) :
    next_state = self.next_state(self.agent_state, action)
    reward = self.reward(self.agent_state, action, next_state)
    done = True if next_state == self.goal_state else False
    self.agent_state = next_state
    return next_state, reward, done