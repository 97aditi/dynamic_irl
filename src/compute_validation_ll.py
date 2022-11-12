import numpy as np
from src.value_iteration_torchversion import time_varying_value_iteration
import torch

def get_validation_ll(seed, P_a, trajectories, hyperparams, a, goal_maps, gamma=0.9):
    """ obtain the ll of the held-out trajectories using a given parameter setting
        args:
            P_a (N_STATES X N_STATES X N_ACTIONS): labyrinth/gridworld transition matrix
            trajectories (list): list of expert trajectories; each trajectory is a dictionary with 'states' and 'actions' as keys.
            hyperparams (list): current setting of hyperparams, of size N_MAPS
            a (array of size N_MAPS x T): current setting of time-varying weights 
            goal_maps(array of size N_MAPS x N_STATES): goal maps columns contains u_e, u_th, u_ho etc
            gamma (float): discount factor
        returns:
           val_ll (float): validation ll of held-out trajectories
    """   

    torch.manual_seed(seed)
    np.random.seed(seed)

    # concatenate expert trajectories
    assert(len(trajectories)>0), "no expert trajectories found!"
    state_action_pairs = []
    for num, traj in enumerate(trajectories):
        states = np.array(traj['states'])[:,np.newaxis]
        actions = np.array(traj['actions'])[:,np.newaxis]
        if len(states) == len(actions)+1:
            states = np.array(traj['states'][:-1])[:,np.newaxis]
        assert len(states) == len (actions), "states and action sequences dont have the same length"
        T = len(states)
        state_action_pairs_this_traj = np.concatenate((states, actions), axis=1)
        assert state_action_pairs_this_traj.shape[0]==len(states), "error in concatenation of s,a,s' tuples"
        assert state_action_pairs_this_traj.shape[1]==2, "states and actions are not integers?"
        state_action_pairs.append(state_action_pairs_this_traj)

    # converting to tensors
    P_a = torch.from_numpy(P_a)
    a = torch.from_numpy(a)
    goal_maps = torch.from_numpy(goal_maps)
    sigmas = torch.tensor(hyperparams)
    N_STATES = P_a.shape[0]
    N_MAPS = a.shape[0]

    log_likelihood = getLL(goal_maps, state_action_pairs, hyperparams, a, P_a, gamma)
        
    return log_likelihood.detach().numpy()



def getLL(goal_maps, state_action_pairs, hyperparams, a, P_a, gamma):
    """ returns  likelihood at given goal_maps
        args:
            same as neglogll
        returns:
            log_likelihood summed over all the state action terms 
    """

    T = state_action_pairs[0].shape[0]
    N_STATES = P_a.shape[0]
    N_MAPS = a.shape[0]
     
    assert(goal_maps.shape[0]==N_MAPS and goal_maps.shape[1]==N_STATES), "goal maps are not of the appropriate shape"

    # ------------------------------------------------------------------
    # compute the likelihood terms 
    # ------------------------------------------------------------------
    # this requires computing time-varying policies, \pi_t, and obtaining log pi_t(a_t|s_t)
    # compute rewards for every time t first
    rewards = a.T@goal_maps
    assert rewards.shape[0]==T and rewards.shape[1]==N_STATES,"rewards not computed correctly"
    # policies should be T x N_STATES X N_ACTIONS
    values, _, log_policies = time_varying_value_iteration(P_a, rewards=rewards, gamma=gamma, error=0.1, return_log_policy=True)

    # compute the ll for all trajectories
    num_trajectories = len(state_action_pairs)
    log_likelihood = 0
    for i in range(num_trajectories):
        states, actions = torch.tensor(state_action_pairs[i][:,0], dtype=torch.long), torch.tensor(state_action_pairs[i][:,1], dtype=torch.long)
        log_likelihood = log_likelihood + torch.sum(log_policies[range(T), states, actions])

    return log_likelihood
