import torch
from torch import logsumexp

def time_varying_value_iteration(P_a, rewards, gamma, error=0.01, return_log_policy=False):
    """
    time-varying soft value iteration function (to ensure that the policy is differentiable)

    inputs:
    P_a         N_STATESxN_STATESxN_ACTIONS transition probabilities matrix - 
                                P_a[s0, s1, a] is the transition prob of 
                                landing at state s1 when taking action 
                                a at state s0
    rewards     T X N_STATES matrix - rewards for all the states
    gamma       float - RL discount
    error       float - threshold for a stop

    returns:
    values    T X N_STATES matrix - estimated values
    policy    T X N_STATES x N_ACTIONS matrix - policy
    """
    N_STATES, _, N_ACTIONS = P_a.shape
    T = rewards.shape[0]
    values = torch.zeros([T, N_STATES], requires_grad=True)
    P_a = P_a.type(torch.float32)
    rewards = rewards.type(torch.float32)

    # estimate values and q-values iteratively
    while True:
        values_tmp = values
        # this is T X N_STATES X N_ACTIONS 
        q_values = torch.stack([rewards + gamma*(values_tmp@P_a[:,:,a].T) for a in range(N_ACTIONS)])
        
        q_values = torch.transpose(q_values, 0, 1)
        q_values = torch.transpose(q_values, 1, 2)

        assert q_values.shape[0]==T and q_values.shape[1]==N_STATES, "q-values don't have the appropriate dimensions"
        assert q_values.shape[2]==N_ACTIONS, "q-values don't have the appropriate dimensions"
        values = logsumexp(q_values, axis=2)
        
        if torch.max(torch.abs(values - values_tmp)) < error:
            break

    # generate policy
    log_policy = q_values - values[:,:,None]
    policy = torch.exp(log_policy)
    if return_log_policy:
        return values, policy, log_policy
    else:
        return values, policy