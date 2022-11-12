import numpy as np


class GridWorld(object):
  """
  Grid world environment
  """

  def __init__(self, grid, terminals, trans_prob=1):
    """
    input:
      grid        2-d list of the grid including the reward
      terminals   a set of all the terminal states
      trans_prob  transition probability when given a certain action
    """
    self.height = len(grid)
    self.width = len(grid[0])
    self.n_states = self.height*self.width
    for i in range(self.height):
      for j in range(self.width):
        grid[i][j] = str(grid[i][j])


    self.terminals = terminals
    self.grid = grid
    self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
    self.actions = [0, 1, 2, 3, 4]
    self.n_actions = len(self.actions)
    self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
    self.trans_prob = trans_prob


  def get_reward(self, state):
    """
    returns
      the reward on current state
    """
    if not self.grid[state[0]][state[1]] == 'x':
      return float(self.grid[state[0]][state[1]])
    else:
      return 0


  def get_transition_states_and_probs(self, state, action):
    """
    get all the possible transition states and their probabilities with [action] on [state]
    args
      state     (y, x)
      action    int
    returns
      a list of (state, probability) pair
    """
    if self.is_terminal(tuple(state)):
      return [(tuple(state), 1)]

    if self.trans_prob == 1:
      inc = self.neighbors[action]
      nei_s = (int(state[0] + inc[0]), int(state[1] + inc[1]))
      if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[
              1] >= 0 and nei_s[1] < self.width and self.grid[nei_s[0]][nei_s[1]] != 'x':
        return [(nei_s, 1)]
      else:
        # if the state is invalid, stay in the current state
        return [(state, 1)]
    else:
      mov_probs = np.zeros([self.n_actions])
      mov_probs[action] = self.trans_prob
      mov_probs += (1-self.trans_prob)/self.n_actions

      for a in range(self.n_actions):
        inc = self.neighbors[a]
        nei_s = (int(state[0] + inc[0]), int(state[1] + inc[1]))
        if nei_s[0] < 0 or nei_s[0] >= self.height or \
           nei_s[1] < 0 or nei_s[1] >= self.width or self.grid[nei_s[0]][nei_s[1]] == 'x':
          # if the move is invalid, accumulates the prob to the current state
          mov_probs[self.n_actions-1] += mov_probs[a]
          mov_probs[a] = 0

      res = []
      for a in range(self.n_actions):
        if mov_probs[a] != 0:
          inc = self.neighbors[a]
          nei_s = (int(state[0] + inc[0]), int(state[1] + inc[1]))
          res.append((nei_s, mov_probs[a]))
      return res


  def is_terminal(self, state):
    """
    returns
      True if the [state] is terminal
    """
    if tuple(state) in self.terminals:
      return True
    else:
      return False


  def reset(self, start_pos):
    """
    Reset the gridworld for model-free learning. It assumes only 1 agent in the gridworld.
    args
      start_pos     (i,j) pair of the start location
    """
    self._cur_state = start_pos


  def get_current_state(self):
    return self._cur_state


  def step(self, action):
    """
    Step function for the agent to interact with gridworld
    args
      action        action taken by the agent
    returns
      current_state current state
      action        input action
      next_state    next_state
      reward        reward on the next state
      is_done       True/False - if the agent is already on the terminal states
    """
    if self.is_terminal(self._cur_state):
      self._is_done = True
      return self._cur_state, action, self._cur_state, self.get_reward(self._cur_state), True

    st_prob = self.get_transition_states_and_probs(self._cur_state, action)

    sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
    last_state = self._cur_state
    next_state = st_prob[sampled_idx][0]
    reward = self.get_reward(last_state)
    self._cur_state = next_state
    return last_state, action, next_state, reward, False


  def get_transition_mat(self):
    """
    get transition dynamics of the gridworld
    return:
      P_a         NxNxN_ACTIONS transition probabilities matrix -
                    P_a[s0, s1, a] is the transition prob of
                    landing at state s1 when taking action
                    a at state s0
    """
    N_STATES = self.height*self.width
    N_ACTIONS = len(self.actions)
    P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    for si in range(N_STATES):
      posi = self.idx2pos(si)
      for a in range(N_ACTIONS):
        probs = self.get_transition_states_and_probs(posi, a)

        for posj, prob in probs:
          sj = self.pos2idx(posj)
          # Prob of si to sj given action a
          P_a[si, sj, a] = prob
    return P_a


  def pos2idx(self, pos):
    """
    input:
      column-major 2d position
    returns:
      1d index
    """
    return pos[0] + pos[1] * self.height


  def idx2pos(self, idx):
    """
    input:
      1d idx
    returns:
      2d column-major position
    """
    return (int(idx % self.height), int(idx / self.height))