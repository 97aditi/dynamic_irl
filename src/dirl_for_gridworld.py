import argparse, os
import numpy as np
import pickle
from src.optimize_weights import getMAP_weights
from src.optimize_goal_maps import getMAP_goalmaps
from src.compute_validation_ll import get_validation_ll


def fit_dirl_gridworld(num_trajs, lr_weights, lr_maps, max_iters, gamma, N_MAPS, seed, GEN_DIR_NAME, REC_DIR_NAME):
    """ fits DIRL on simulated trajectories from the gridworld environment given hyperparameters
        and saves all recovered parameters

        args:
        version (int): choose which version of the simulated trajectories to use
        num_trajs (int): choose how many trajectories to use
        lr_weights (float): choose learning rate for weights
        lr_maps (float): choose learning rate for goal maps
        max_iters (int): num iterations to run the optimization for weights/goal maps per outer loop of dirl
        gamma (float): value iteration discount parameter
        N_MAPS (int): # of goal maps to use while fitting DIRL
        seed (int): initialization seed
        GEN_DIR_NAME (str): name of the folder that contains the trajectories and generative parameters
        REC_DIR_NAME (str): name of the folder to store recovered parameters
    """
    np.random.seed(seed)
    # load the files to obtain the simulated trajectories from
    file = open(GEN_DIR_NAME + '/generative_parameters.pickle', 'rb')
    file_trajs = open(GEN_DIR_NAME +'/expert_trajectories.pickle', 'rb')

    # create folder to store recovered parameters
    save_dir = REC_DIR_NAME + "/maps_"+str(N_MAPS)+ "_lr_" + str(lr_maps)+ "_"+str(lr_weights)

    # check if save_dir exists, else create it 
    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir, exist_ok = True)

    # load expert trajs
    all_expert_trajectories = pickle.load(file_trajs)
    # slice to only the # of trajs that we need
    all_expert_trajectories = all_expert_trajectories[:num_trajs]
    print("Loaded "+str(num_trajs)+" expert trajectories for gridworld!", flush=True)
    T = len(all_expert_trajectories[0]["actions"])
    print("Using "+str(T)+" state-action pairs per trajectory.", flush=True)

    # split into train and val sets
    val_indices = np.arange(start=0, stop=num_trajs, step=5)
    train_indices = np.delete(np.arange(num_trajs), val_indices)
    val_expert_trajectories = [all_expert_trajectories[val_idx] for val_idx in val_indices]
    expert_trajectories = [all_expert_trajectories[train_idx] for train_idx in train_indices]
    print("# of validation trajs: " +str(len(val_expert_trajectories)))
    print("# of training trajs: " +str(len(expert_trajectories)))

    # loading some relevant generative parameters
    generative_params = pickle.load(file)
    P_a = generative_params['P_a'] # transition matrix
    N_STATES = P_a.shape[0] # no of states in gridworld
    sigma = generative_params['sigmas'][0] # noise covariance of time-varying weights
    sigmas = [sigma]* N_MAPS

    # choose a random initial setting for the weights (parameters)
    weights = (np.random.multivariate_normal(mean=np.zeros(T,), cov = sigmas[0]*np.eye(T,), size=N_MAPS)).reshape((N_MAPS,T))
    # choose a random initial setting for the goal maps (parameters)
    goal_maps = np.random.uniform(size=(N_MAPS,N_STATES))

    # save things
    rec_weights = []
    rec_goal_maps = []
    losses_all_weights = []
    losses_all_maps = []
    val_lls = []

    for i in range(20):
        print("At iteration: "+str(i), flush=True)
        print("-------------------------------------------------", flush=True)
        # get the MAP estimates of time-varying weights and list of losses at every time step
        a_MAPs, losses =  getMAP_weights(seed, P_a, expert_trajectories, hyperparams = sigmas, goal_maps = goal_maps, 
                                                        a_init=weights, max_iters=max_iters, lr=lr_weights, gamma=gamma)
        weights = a_MAPs[-1]
        rec_weights.append(weights)
        losses_all_weights = losses_all_weights + losses

        # save recovered time-varying weights as well as training loss
        np.save(save_dir + "/weights_trajs_"+str(num_trajs)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", rec_weights)
        np.save(save_dir + "/losses_weights_trajs_"+str(num_trajs)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", losses_all_weights)

        # get the optimal estimates of the goal maps and list of losses at every time step
        goal_maps_MLEs, losses =  getMAP_goalmaps(seed, P_a, expert_trajectories, hyperparams = sigmas, a=weights, 
                                                        goal_maps_init = goal_maps, max_iters=max_iters, lr=lr_maps, gamma=gamma)
        goal_maps = goal_maps_MLEs[-1]
        rec_goal_maps.append(goal_maps)
        losses_all_maps = losses_all_maps + losses

        # save recovered goal maps as well as training loss
        np.save(save_dir + "/goal_maps_trajs_"+str(num_trajs)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", rec_goal_maps)
        np.save(save_dir + "/losses_maps_trajs_"+str(num_trajs)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", losses_all_maps)

        val_ll = get_validation_ll(seed, P_a, val_expert_trajectories, hyperparams = sigmas, a=weights, goal_maps=goal_maps, gamma=gamma)
        val_lls.append(val_ll)
        # save validation LL on held-out trajectories
        np.save(save_dir + "/validation_lls_"+str(num_trajs)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", val_lls) 




