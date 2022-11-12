# Script to generate data for continuously varying reward parameters
# Given a set of hyperparameters, first simulate a reward trajectory using these hyperparameters
# Next, generate trajectory data within the gridworld environment
import os
import numpy as np
import matplotlib.pyplot as plt
from envs import gridworld
from collections import namedtuple
import pickle
from plot_utils.plot_simulated_data_gridworld import plot_rewards_all, plot_gridworld_trajectories
from value_iteration import time_varying_value_iteration

np.random.seed(50)

Step = namedtuple('Step','cur_state action next_state reward done')

def create_goal_maps(num_gridworld_states, LOCATION_WATER, LOCATION_HOME):
    '''
    create goal reward maps
    '''
    home_map = np.zeros((num_gridworld_states))
    home_map[LOCATION_HOME] = 1

    water_map = np.zeros((num_gridworld_states))
    water_map[LOCATION_WATER] = 1

    goal_maps = np.array([home_map, water_map])
    return goal_maps


def generate_weight_trajectories(sigmas, weights0, T):
    '''Simulates time varying weights, for a given sigmas array

    Args:
        sigma: array of length K, the smoothness of each reward weight
        weights0: values of time-varying weights parameters at t=0
        T: length of trajectory to generate (i.e. number of states visited in the gridworld)

    Returns:
        rewards: array of KxT reward parameters
    '''
    K = len(sigmas)
    noise = np.random.normal(scale=sigmas, size=(T, K))
    # home port
    np.random.seed(50)
    noise[:,0] = np.random.normal(0.01, scale=sigmas[0], size=(T,))
    # water port
    np.random.seed(100)
    noise[:,1] = np.random.normal(-0.02, scale=sigmas[1], size=(T,))
    noise[0,:] = weights0
    weights = np.cumsum(noise, axis=0)
    return weights #array of size (TxK)


def generate_expert_trajectories(grid_H, grid_W, time_varying_rewards, time_varying_policy, T,  GAMMA):
    '''
    given reward trajectories, generate state-action trajectories, assuming that these agents act optimally
    under the provided reward function

    returns:
    trajectory  - a list of Steps representing an episode
    '''

    # initial reward map
    r_map = np.reshape(np.array(time_varying_rewards[:,0]), (grid_H,grid_W), order='F')
    gw = gridworld.GridWorld(r_map, {},)
    start_pos = (np.random.randint(0, gw.height), np.random.randint(0, gw.width))
    gw.reset(start_pos)
    cur_state = start_pos #current state
    states = [gw.pos2idx(cur_state)] # save states in their rolled out 1d rep from 1 to num_states
    states2d = [cur_state] # save x,y coordinates of states
    actions = []
    rewards = []
    dones = []
    for t in range(0, T-1):
        gw.grid = np.reshape(np.array(time_varying_rewards)[:,t], (grid_H,grid_W), order='F') #update the reward function of the environment
        policy_t = time_varying_policy[t] # get policy
        action = np.random.choice(range(policy_t.shape[1]), p=policy_t[gw.pos2idx(cur_state)]) # take action
        cur_state, action, next_state, reward, is_done = gw.step(action) # update current state
        states.append(gw.pos2idx(next_state))
        states2d.append(next_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(is_done)
        if is_done:
            break
    # create a trajectory dict
    traj = {'states': np.array(states), 'states2d': np.array(states2d),
            'actions': np.array(actions), 'rewards': np.array(rewards),
            'dones': np.array(dones)}
    return traj


def get_time_varying_policy(rewards, P_a, gamma):
    '''
    given rewards, obtain time-varying optimal policy
    '''
    time_varying_values, time_varying_policy = time_varying_value_iteration(P_a, rewards.T, gamma, error=0.001)
    return time_varying_policy, time_varying_values


def main(gridworld_H, gridworld_W, N_experts, T,
         LOCATION_WATER, LOCATION_HOME, VERSION = 1, GAMMA=0.9, plot_data=False):
    '''
    generate time-varying weights on goal maps, and then generate expert
    trajectories
    '''

    # select number of maps
    num_maps = 2

    # choose noise covariance for the random walk priors over weights corresponding to these maps
    sigmas = [2**-(3.5)]*num_maps

    weights0 = np.zeros(num_maps) #initial weights at t=0
    weights0[1] = 1

    goal_maps = create_goal_maps(gridworld_H*gridworld_W, LOCATION_WATER, LOCATION_HOME)
    #size is num_maps x num_states

    #generate time-varying weights
    time_varying_weights = generate_weight_trajectories(sigmas,
                                                        weights0,
                                                        T) #T x num_maps
    # now obtain time-varying reward maps
    rewards = time_varying_weights@goal_maps #array of size Txnum_states
    rewards = rewards.T

    r_map = np.reshape(np.array(rewards[:, 0]), (gridworld_H, gridworld_W), order='F')
    gw = gridworld.GridWorld(r_map, {},)  # instantiate
    # gridworld environment.  {} indicates that there are no terminal states
    P_a = gw.get_transition_mat()
    time_varying_policy, time_varying_values = get_time_varying_policy(rewards, P_a, GAMMA)

    trajs_all_experts = []
    for expert in range(N_experts):
        traj = generate_expert_trajectories(gridworld_H, gridworld_W, rewards, time_varying_policy, T, GAMMA,)
        trajs_all_experts.append(traj)

    generative_parameters = {}
    generative_parameters['P_a'] = P_a
    generative_parameters['generative_rewards'] = rewards
    generative_parameters['goal_maps'] = goal_maps
    generative_parameters['time_varying_weights'] = time_varying_weights
    generative_parameters['sigmas'] = sigmas
    generative_parameters['time_varying_policy'] = time_varying_policy

    # now save everything
    save_dir = '../data/simulated_gridworld_data/exclude_explore_share_weights_'+str(VERSION)
    # check if save_dir exists, else create it
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok = True)

    with open(
            save_dir+
            '/expert_trajectories.pickle',
                'wb') as handle:
        pickle.dump(trajs_all_experts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
            save_dir+
            '/generative_parameters.pickle', 'wb') as handle:
        pickle.dump(generative_parameters, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    if plot_data:
        plot_rewards_all(goal_maps, time_varying_weights,
                                    rewards, gridworld_H, gridworld_W,LOCATION_WATER, LOCATION_HOME,
                                    save_name=save_dir+'/generative_rewards.png')
        # plot a few example trajectories:
        for i in [0, 10, 20]:
            traj = trajs_all_experts[i]
            fig, ax = plt.subplots()
            plot_gridworld_trajectories(gridworld_H, gridworld_W, traj, ax)
            fig.savefig(
                save_dir+
                '/expert_traj_' + str(
                    i) + '.png')



if __name__ == "__main__":
   
    # set height and width of the gridworld
    gridworld_H, gridworld_W = 5, 5
    num_gridworld_states = gridworld_H*gridworld_W
    N_experts = 200 #number of trajectories to generate
    T = 50  #length of each trajectory

    # set locations of home and water port
    LOCATION_WATER = 14
    LOCATION_HOME = 0

    # set a version number for saving
    VERSION = 1

    main(gridworld_H, gridworld_W, N_experts, T,
        LOCATION_WATER, LOCATION_HOME, VERSION, GAMMA=0.9, plot_data=True)
