import numpy as np

class LabyrinthEnv():
    """ creates an environment to replicate the labyrinth task from Rosenberg et al. 2021; using an additional stay action """

    def __init__(self, reward_state=100, n_states=127, max_episode_length=500, reward_map = None):
        # space of all actions (0: left, 1: right, 2: reverse, 3: stay)
        self.action_space = [0,1,2,3]
        self.n_actions = len(self.action_space) 
        # space of all observations 
        self.n_states = n_states
        # a tuple containing max and min rewards
        self.reward_range = (0,1)
        # max steps in an episode
        self.max_episode_length = max_episode_length
        # length of the current episode
        self.length_of_episode = 0
        # specifying home state
        self.home_state = 0
        # specifying water port (end of maze, i.e. the reward state; assuming reward is provided in [0, N_STATES]))
        self.water_port = reward_state 
        # initial condition
        self.state = None
        # terminal states start from 
        self.terminal_start = int(n_states/2) 
        # reward map if rewards are provided externally
        self.reward_map = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, reset() is called.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            current_state (int): present state of the environment
            action (int): action taken by the agent
            next_state (int): next state in the environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        """

        current_state = self.state 
        done = False
        reward = 0

        # reverse action
        if action==2:
            if (current_state == self.home_state):
                self.state = current_state
            else:
                self.state = self.take_action(action, current_state)
        
        # left action
        if action==0:
            # handling terminal states
            if current_state>=self.terminal_start:
                self.state = current_state
            else:
                self.state = self.take_action(action, current_state)
        
        # right action
        if action==1:
            # handling terminal states
            if current_state>=self.terminal_start:
                self.state = current_state
            else:
                self.state = self.take_action(action, current_state)

        # stay action
        if action==3:
            self.state = current_state

        # computing rewards and checking if episode has come to an end
        if self.reward_map is None:
            if self.state == self.water_port:
                # reward for arriving at the water port
                reward = 1
        else:
            reward = self.reward_map[int(self.state)]

        self.length_of_episode = self.length_of_episode+1
        if self.length_of_episode >= self.max_episode_length:
            done = True

        return current_state, action, self.state, reward, done

    def reset(self, state):
        """Resets the environment to an initial state and returns an initial
        observation.
        This method should also reset the environment's random number
        generator(s) if `seed` is an integer or if the environment has not
        yet initialized a random number generator. 
        Returns:
            observation (object): the initial observation.
        """
        # setting initial state to provided state
        self.state = state
        # setting length of episode to 0
        self.length_of_episode = 0
        return self.state


    def take_action(self, action, state):
        """ helper function to compute the next state given a state and action
        Returns:
            observation (object): the next observation
        """
        # adding 1 for ease of computation
        state = state + 1
        # taking left action: shift by one position to left in binary rep
        if action==0:
            new_state = state*2

        # taking right action: shift by one position to left and convert last digit to 1
        if action==1:
            new_state = state*2 + 1

        # taking reverse action: shift by one position to right
        if action==2:
            new_state = int(state/2)

        # subtracting 1 to bring it back to [0,...N_STATES]
        return new_state-1


    def get_current_state(self):
        return self.state

    def get_transition_mat(self):
        """ get transition matrix of the labyrinth
        returns:
        P_a: n_state x n_state x n_actions such that P(s0,s1,a) is the prob of going from s0 to s1 using a"""

        P_a = np.zeros((self.n_states, self.n_states, self.n_actions))
        # P_a is indexed from 0, so every state is going to be i+1
        # action space: (0: left, 1: right, 2: reverse, 3:stay)
        for i in range(self.n_states):
            # the stay action now allows us to stay at the same state
            P_a[i,i,3] = 1
            # check for home state:
            if i==0:
                # left action
                P_a[i,1,0] = 1
                # right action 
                P_a[i,2,1] = 1
                # reverse action should lead to the state itself
                P_a[i,i,2] = 1
            # checking terminal states
            elif i>=self.terminal_start:
                # adding 1 to bring states to the range 1..N_STATES which is easier for computing next states
                state = i+1
                reverse_state = int(state/2)
                # only reverse action is feasible
                P_a[i, reverse_state-1, 2] = 1
                # left and right actions should lead to the state itself
                P_a[i, i, 0] = 1
                P_a[i, i, 1] = 1
            else:
                state = i+1
                # left action
                left_state = 2*(state)
                P_a[i,left_state-1,0] = 1
                # right action 
                right_state = 2*(state)+1
                P_a[i,right_state-1,1] = 1
                # reverse action
                reverse_state = int((state)/2) 
                P_a[i, reverse_state-1, 2] = 1

        return P_a
