import torch
import numpy as np
from src.helpers import Dv_torch
from src.value_iteration_torchversion import time_varying_value_iteration
from torch.autograd.functional import hessian

def compute_inv_hessian(seed, P_a, trajectories, hyperparams, a, goal_maps, gamma):
    """ returns the inverse hessian of the neglogpost w.r.t. the weights
        args:
            P_a (N_STATES X N_STATES X N_ACTIONS): labyrinth/gridworld transition matrix
            trajectories (list): list of expert trajectories; each trajectory is a dictionary with 'states' and 'actions' as keys.
            hyperparams (list): current setting of hyperparams, of size N_MAPS
            a (array of size N_MAPS x T): time-varying weights
            goal maps (array of size N_MAPS x N_STATES): columns contains the goal maps
            gamma (float): discount factor for value iteration
        returns:
            inv_hess: inverse of the hessian computed at provided goal maps
    """   

    torch.manual_seed(seed)
    np.random.seed(seed)

    def neglogll(a):

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
        # blocked difference matrix : Dv; computing Dv@theta_flat
        E_flat = Dv_torch(a, N_MAPS) 
        # calculating the log prior
        log_prior = (1 / 2) * (logdet_invSigma - (E_flat**2 * invSigma_diag).sum())

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

        negL = -log_likelihood - log_prior
        return negL

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
    a = torch.from_numpy(a)
    sigmas = torch.tensor(hyperparams)
    N_STATES = P_a.shape[0]
    N_MAPS = goal_maps.shape[0]
    assert goal_maps.shape[1]==N_STATES, "goal maps should be tensors with length as no. of states"
    assert a.shape[0]==N_MAPS and a.shape[1]==T, "weights should have weights N_MAPS X T"

    
    # flatten them to pass as arguments to the neglogpost function
    a_flat = a.flatten()
    # compute hessian using torch
    hess = hessian(neglogll, inputs=(a_flat))

    # check if the hessian is symmetric
    assert(torch.allclose(hess.T, hess, rtol=0.1, atol=0.1)), "hessian should be symmetric"
    # this is really just to check if the hessian is PD
    L = torch.linalg.cholesky(hess)
    # invert this now
    inv_hess = torch.linalg.inv(hess)

    return inv_hess.detach().numpy()


def compute_conf_interval(inv_hess, T, N_MAPS, N_STATES):
    """ computes confidence interval for weights and goal maps given the inverse hessian """
    assert inv_hess.shape[0]==N_MAPS*T and inv_hess.shape[1]==N_MAPS*T, "inverse hessian has the wrong shape" 
    
    diag_inv_hess = np.diag(inv_hess)
    assert(np.all(diag_inv_hess>0)), "the hessian is not computed at a minima"
    
    # confidence interval for the weights
    conf_weights = np.sqrt(diag_inv_hess)
    # reshape to N_MAPS X T
    conf_weights = conf_weights.reshape((N_MAPS, T))
    
    return conf_weights

