import numpy as np
from src.value_iteration_torchversion import time_varying_value_iteration
from src.helpers import Dv_torch
import torch

def getMAP_weights(seed, P_a, trajectories, goal_maps, hyperparams, a_init=None, max_iters = 500, lr = 0.01, gamma=0.9, info={'Neval': 0}):
    """ obtain the MAP estimates of model parameters
        args:
            P_a (N_STATES X N_STATES X N_ACTIONS): labyrinth/gridworld transition matrix
            trajectories (list): list of expert trajectories; each trajectory is a dictionary with 'states' and 'actions' as keys.
            hyperparams (list): current setting of hyperparams, of size N_MAPS
            goal maps (array of size N_MAPS x N_STATES): columns contains u_e, u_th, u_ho etc
            a_init (array of size N_MAPS x T): initial guess for a (T: total # of state-action pairs across trajectories)
            max_iters (int): number of SGD iterations to optimize this for
            lr (float): learning rate
            gamma (float): discount factor in value iteration
            info: dict with anything that we'd like to store for printing purposes
        returns:
            a_MAP (3-d array: (max_iters/10) x N_MAPS x T): MAP estimates of the time-varying weghts saved after every 10 iterations
            losses (list): values of the negative log posterior after every iteration
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
    goal_maps = torch.from_numpy(goal_maps)
    sigmas = torch.tensor(hyperparams)
    N_STATES = P_a.shape[0]
    N_MAPS = goal_maps.shape[0]
    assert goal_maps.shape[1]==N_STATES, "goal maps should be tensors with length as no. of states"

    # initial value of time-varying weights
    if a_init is None:
        a_init = torch.zeros(T*N_MAPS, requires_grad = True)
    else:
        assert a_init.shape[0]==N_MAPS and a_init.shape[1]==T, "initialize weights as N_MAPS X T"
        a_init = torch.from_numpy(a_init).flatten()
        a_init.requires_grad = True

    print("Minimizing the negative log posterior ...")
    print('{0} {1}'.format('# n_iters', 'neg LP'))
    optimizer = torch.optim.Adam([a_init], lr=lr)
    # saving the losses
    losses = []
    # saving MAP estimates after every 10 iterations
    a_MAPs = []

    for i in range(max_iters):
        loss = neglogpost(a_init, state_action_pairs, sigmas, goal_maps, P_a,  gamma, info)
        losses.append(loss.detach().numpy())
        # taking gradient step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%10 == 0 or i==max_iters-1:
            a_MAP = a_init.detach().numpy()
            a_MAP = np.reshape(a_MAP, (N_MAPS, T))
            a_MAPs.append(a_MAP.copy())

    return a_MAPs, losses


def neglogpost(a, state_action_pairs, hyperparams, goal_maps, P_a, gamma, info):
    '''Returns negative log posterior 
        args:
            a (1-d tensor: T*N_MAPS)
            state_action_pairs (list of len(trajectories), with each element an array: T x (STATE_DIM + ACTION_DIM ))
            hyperparams (tensor): current setting of hyperparams, contains key 'sigmas' whick is array of size 3 with elements \sigma_e, \sigma_th, \sigma_ho
            goal maps (tensor of size N_MAPS x N_STATES): columns contains u_e, u_th, u_ho
            P_a (tensor: N_STATES X N_STATES X N_ACTIONS): transition matrix 
            gamma (float): discount factor in value iteration
            info: dict with anything that we'd like to store for printing purposes
        returns:
            negL : negative log posterior
    '''
    
    num_trajectories = len(state_action_pairs)
    log_prior, log_likelihood = getPosterior(a, state_action_pairs, hyperparams, goal_maps, P_a, gamma)
    negL = (-log_prior-log_likelihood)/num_trajectories

    info['Neval'] = info['Neval']+1
    n_eval = info['Neval']


    print('{0}, {1}'.format(n_eval, negL))
    return negL

def getPosterior(a, state_action_pairs, hyperparams, goal_maps, P_a,  gamma):
    """ returns prior and likelihood at given time-varying weights and goal maps
        args:
            same as neglogpost
        returns:
            log_prior: log prior of time-varying weights
            log_likelihood summed over all the state action terms 
    """

    T = state_action_pairs[0].shape[0]
    N_STATES = P_a.shape[0]
    N_MAPS = goal_maps.shape[0]
     
    assert(a.shape[0]==T*N_MAPS), "time-varying weights are not of the appropriate shape"

    # ----------------------------------------------------------------
    # construct random-walk prior, calculate prior
    # ----------------------------------------------------------------
    # diagonal of inverse of the sigma matrix
    invSigma_diag = torch.zeros((T*N_MAPS))
    sigmas = hyperparams
    for s in range(N_MAPS):
        sigma_s =sigmas[s]
        invSigma_diag[s*T:(s+1)*T] = 1/(sigma_s**2)
    logdet_invSigma = torch.sum(torch.log(invSigma_diag))
    # blocked difference matrix : Dv; computing Dv@a_flat
    E_flat = Dv_torch(a, N_MAPS) 
    # calculating the log prior
    logprior = (1 / 2) * (logdet_invSigma - (E_flat**2 * invSigma_diag).sum())

    # ------------------------------------------------------------------
    # compute the likelihood terms 
    # ------------------------------------------------------------------
    # this requires computing time-varying policies, \pi_t, and obtaining log pi_t(a_t|s_t)
    a_reshaped = a.reshape(N_MAPS, -1)
    # compute rewards for every time t first
    rewards = a_reshaped.T@goal_maps
    assert rewards.shape[0]==T and rewards.shape[1]==N_STATES,"rewards not computed correctly"
    # policies should be T x N_STATES X N_ACTIONS
    values, _, log_policies = time_varying_value_iteration(P_a, rewards=rewards, gamma=gamma, error=0.1, return_log_policy=True)
    # compute the ll for all trajectories
    num_trajectories = len(state_action_pairs)
    log_likelihood = 0
    for i in range(num_trajectories):
        states, actions = torch.tensor(state_action_pairs[i][:,0], dtype=torch.long), torch.tensor(state_action_pairs[i][:,1], dtype=torch.long)
        log_likelihood = log_likelihood + torch.sum(log_policies[range(T), states, actions])

    return logprior, log_likelihood