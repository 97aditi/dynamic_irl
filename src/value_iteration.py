import numpy as np
from scipy.special import logsumexp

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
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  # no. of time steps
  T = rewards.shape[0]

  values = np.zeros([T, N_STATES])
  q_values = np.zeros([T, N_STATES, N_ACTIONS])

  # estimate values and q-values iteratively
  while True:
    values_tmp = values.copy()
    q_values = np.stack([rewards + sum([gamma* (np.outer(P_a[:, s1, a], values_tmp[:,s1])).T
                                                for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
    q_values = np.transpose(q_values, (1, 2, 0))
    assert q_values.shape[0]==T and q_values.shape[1]==N_STATES, "q-values don't have the appropriate dimensions"
    assert q_values.shape[2]==N_ACTIONS, "q-values don't have the appropriate dimensions"
    values = logsumexp(q_values, axis=2)
    if max([abs(values[t,s] - values_tmp[t,s]) for s in range(N_STATES) for t in range(T)]) < error:
      break

  # generate policy
  log_policy = q_values - values[:,:,np.newaxis]
  policy = np.exp(log_policy)

  if return_log_policy:
    return values, policy, log_policy
  else:
    return values, policy


