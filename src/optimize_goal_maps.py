import numpy as np
from src.value_iteration_torchversion import time_varying_value_iteration
import torch

def getMAP_goalmaps(seed, P_a, trajectories, hyperparams, a, goal_maps_init = None, max_iters = 500, lr = 0.01, gamma=0.9, lam=0, info={'Neval': 0}):
    """ obtain the MAP estimates of model parameters
        args:
            seed (int); initialization seed
            P_a (N_STATES X N_STATES X N_ACTIONS): labyrinth/gridworld transition matrix
            trajectories (list): list of expert trajectories; each trajectory is a dictionary with 'states' and 'actions' as keys.
            hyperparams (list): current setting of sigmas, of size [N_MAPS,]
            a (array of size N_MAPS x T): current setting of time-varying weights 
            goal_maps_init (array of size N_MAPS x N_STATES): intial guess for goal maps 
            max_iters (int): number of SGD iterations to optimize this for
            lr (float): learning rate
            gamma (float): discount factor in value iteration
            lam (float): l2 coefficient 
            info: dict with anything that we'd like to store for printing purposes
        returns:
            goal_maps_MLE (3-d array: (max_iters/10) x N_MAPS x N_STATES): MLE estimates of goal maps after every 10 iterations
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
    a = torch.from_numpy(a)
    sigmas = torch.tensor(hyperparams)
    N_STATES = P_a.shape[0]
    N_MAPS = a.shape[0]

    # initial value of goal maps
    if goal_maps_init is None:
        goal_maps_init = torch.zeros(N_MAPS*N_STATES, requires_grad = True)
    else:
        goal_maps_init = torch.from_numpy(goal_maps_init).flatten()
        goal_maps_init.requires_grad = True

    print("Minimizing the negative log likelihood ...")
    print('{0} {1}'.format('# n_iters', 'neg LL'))
    optimizer = torch.optim.Adam([goal_maps_init], lr=lr)
    # saving the losses
    losses = []
    # saving MLE estimates after every 10 iterations
    goal_maps_MLEs = []


    for i in range(max_iters):
        # l2 prior
        loss_prior = lam*torch.sum(goal_maps_init**2)
        # adding this to the loss
        loss = neglogll(goal_maps_init, state_action_pairs, sigmas, a, P_a, gamma, info) + loss_prior
        losses.append(loss.detach().numpy())
        # taking gradient step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%10 == 0 or i==max_iters-1:
            goal_map_MLE = goal_maps_init.detach().numpy()
            goal_map_MLE = np.reshape(goal_map_MLE, (N_MAPS, N_STATES))
            goal_maps_MLEs.append(goal_map_MLE.copy())
        
    return goal_maps_MLEs, losses


def neglogll(goal_maps, state_action_pairs, hyperparams, a, P_a, gamma, info):
    '''Returns negative log posterior 
        args:
            goal maps (1-d tensor of size N_MAPS*N_STATES)
            state_action_pairs (list of len(trajectories), with each element an array: T x (STATE_DIM + ACTION_DIM ))
            hyperparams (tensor): current setting of hyperparams, contains key 'sigmas' which is array of size N_MAPS
            a (2-d tensor: N_MAPS x T)
            P_a (tensor: N_STATES X N_STATES X N_ACTIONS): transition matrix 
            gamma (float): discount 
            info: dict with anything that we'd like to store for printing purposes
        returns:
            negL : negative log posterior
    '''
    
    num_trajectories = len(state_action_pairs)
    log_likelihood = getLL(goal_maps, state_action_pairs, hyperparams, a, P_a, gamma)
    negL = (-log_likelihood)/num_trajectories

    info['Neval'] = info['Neval']+1
    n_eval = info['Neval']

    print('{0}, {1}'.format(n_eval, negL))
    return negL


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

     
    assert(goal_maps.shape[0]==N_MAPS*N_STATES), "goal maps are not of the appropriate shape"

    # ------------------------------------------------------------------
    # compute the likelihood terms 
    # ------------------------------------------------------------------
    # this requires computing time-varying policies, \pi_t, and obtaining log pi_t(a_t|s_t)
    # compute rewards for every time t first
    goal_maps_reshaped = goal_maps.reshape((N_MAPS,-1))

    rewards = a.T@goal_maps_reshaped
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
