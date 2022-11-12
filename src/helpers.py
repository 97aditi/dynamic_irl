import numpy as np
import torch

def Dv(v, K):
    '''
    Computes D @ v, where D is the blocked difference matrix much more quickly
    '''
    v2 = v.reshape(K, -1)
    v3 = np.hstack((v2[:, 0:1], np.diff(v2, axis=1)))
    v4 = v3.flatten()
    return v4

def Dv_torch(v, K):
    '''
    Computes D @ v, where D is the blocked difference matrix much more quickly
    '''
    v2 = v.reshape(K, -1)
    v3 = torch.hstack((v2[:, 0:1], torch.diff(v2, axis=1)))
    v4 = v3.flatten()
    return v4


def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)