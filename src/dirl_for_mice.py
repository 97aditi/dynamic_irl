import os
import numpy as np
import pickle
from src.envs import labyrinth_with_stay
from src.optimize_weights import getMAP_weights
from src.optimize_goal_maps import getMAP_goalmaps
from src.compute_validation_ll import get_validation_ll

def fit_dirl_mice(water_restricted, lr_weights, lr_maps, max_iters, N_MAPS, sigma, gamma, lamda, seed, GEN_DIR_NAME, REC_DIR_NAME):
    """ fits DIRL on trajectories of mice from Rosenberg et al 2021 given hyperparameters
        and saves all recovered parameters
        args:
            water_restricted (bool): choose whether to train on water-restricted (1) or unresticted mice (0)
            lr_weights (float): choose learning rate for weights
            lr_maps (float): choose learning rate for goal maps
            max_iters (int): number of iterations to run the inner optimization loops for goal maps and weights
            N_MAPS (int): how many goal maps to use
            sigma (float): noise variance for time varying weights
            gamma (float): discount parameter in value iteration
            lamda (float): l2 prior on goal maps
            seed (int): initialization seed
            GEN_DIR_NAME (str): name of the folder that contains the trajectories 
            REC_DIR_NAME (str): name of the folder to store recovered parameters
        """

    np.random.seed(seed)
 
    # load trajectories of mice
    if water_restricted:
        folder_name = GEN_DIR_NAME + 'water_restricted_mice_trajs.pickle'
    else:
        folder_name = GEN_DIR_NAME + 'water_unrestricted_mice_trajs.pickle'
    file = open(folder_name, 'rb')
    all_expert_trajectories = pickle.load(file)
    T = len(all_expert_trajectories[0]["actions"])
    num_trajs = len(all_expert_trajectories)
    print("Loaded "+str(num_trajs)+" trajectories for mice, each trajctory has "+str(T)+" state-action pairs...", flush=True)


    # split trajectories into a training and a validation set
    if water_restricted:
        train_indices = np.load(GEN_DIR_NAME + "restricted_train_indices.npy").astype(int)
        val_indices = np.load(GEN_DIR_NAME + "restricted_val_indices.npy").astype(int)
    else:
        train_indices = np.load(GEN_DIR_NAME + "unrestricted_train_indices.npy").astype(int)
        val_indices = np.load(GEN_DIR_NAME + "unrestricted_val_indices.npy").astype(int)
    val_expert_trajectories = [all_expert_trajectories[val_idx] for val_idx in val_indices]
    expert_trajectories = [all_expert_trajectories[train_idx] for train_idx in train_indices]


    # loading the labyrinth environment with 4 actions per state: reverse, left, right, stay
    lb = labyrinth_with_stay.LabyrinthEnv()
    # loading the transition matrix for this environment 
    P_a = lb.get_transition_mat()
    N_STATES = P_a.shape[0] # num states in this env

    # setting the hyperparameter sigma 
    sigmas = [sigma]*N_MAPS
    

    # choose a random initial setting for the weights (parameters)
    weights = (np.random.multivariate_normal(mean=np.zeros(T,), cov = sigmas[0]*np.eye(T,), size=N_MAPS)).reshape((N_MAPS,T))
    # choose a random initial setting for the goal maps (parameters)
    goal_maps = np.random.uniform(size=(N_MAPS,N_STATES))
    # we already have some information about the two cohorts of mice: 
    # both cohorts of mice spend time at their home state; further water restricted mice are motivated to go to the water port
    # using this information to initialize goal maps
    water_map = np.zeros((N_STATES))
    water_map[100] = 1
    home_map = np.zeros((N_STATES))
    home_map[0] = 1
    # initializing with the above maps
    if N_MAPS==1 and water_restricted:
        goal_maps[0,:] = water_map
    if N_MAPS>=2 and water_restricted:
        goal_maps[0,:] = water_map
        goal_maps[1,:] = home_map
    if water_restricted==0:
        goal_maps[0,:] = home_map
        
    # create a folder to save all recovered parameters                                                                                                                                                                                                                     
    save_dir =  REC_DIR_NAME + "/maps_" +str(N_MAPS)+ "_sigma_" +str(sigmas[0]) + "_lr_" + str(lr_maps)+ \
                                                        "_" + str(lr_weights) +"_"+str(gamma)+"_"+str(lamda)
    # check if save_dir exists, else create it
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok = True)

    # to save things
    rec_weights = []
    rec_goal_maps = []
    losses_all_weights = []
    losses_all_maps = []
    val_lls = []
    for i in range(20):
        print("At iteration: "+str(i), flush=True)
        print("-------------------------------------------------", flush=True)
        # get the MAP estimates and list of losses at every time step
        weights_MAPs, losses =  getMAP_weights(seed, P_a, expert_trajectories, hyperparams = sigmas, goal_maps = goal_maps, 
                                                        a_init=weights, max_iters=max_iters, lr=lr_weights, gamma=gamma)
        weights = weights_MAPs[-1]
        rec_weights.append(weights)
        losses_all_weights = losses_all_weights + losses

        # save recovered time-varying weights as well as training loss
        np.save(save_dir + "/weights_trajs_"+str(num_trajs)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", rec_weights)
        np.save(save_dir + "/losses_weights_trajs_"+str(num_trajs)+"_seed_"+str(seed)+"_iters_"+str(max_iters)+".npy", losses_all_weights)

        # get the MLE estimates and list of losses at every time step
        goal_maps_MLEs, losses =  getMAP_goalmaps(seed, P_a, expert_trajectories, hyperparams = sigmas, a=weights, 
                                                        goal_maps_init = goal_maps, max_iters=max_iters, lr=lr_maps, gamma=gamma, lam = lamda)
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

