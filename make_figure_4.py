import pickle, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from src.envs import labyrinth_with_stay
from plot_utils.MM_Maze_Utils import NewMaze, PlotMazeFunction
from plot_utils.convert_rosenberg_to_our_space import rosenbergnode_to_ournode
from src.compute_conf_interval import compute_conf_interval, compute_inv_hessian
from src.dirl_for_mice import fit_dirl_mice

# matplotlib settings
LEGEND_SIZE = 10
SMALL_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)  # legend fontsize
plt.rcParams['font.sans-serif'] = "Helvetica"
colors_list = ['steelblue', '#D85427', 'tab:green', 'k']

def convert_to_rosenberg_space(values_gt):
    '''
    values_gt is in our space.  In order to plot on top of maze, need to convert to rosenberg space
    '''
    values_gt_rosenberg_space = []
    for i in range(127):
        our_node = rosenbergnode_to_ournode(i)
        values_gt_rosenberg_space.append(values_gt[our_node-1]) # our nodes start at 1 
    return values_gt_rosenberg_space


def compute_scale_and_offset_goal_maps(rec_goal_maps):
    '''
    scale recovered goal maps so that they are in the range of [0,1]
    '''
    N_GOAL_MAPS = rec_goal_maps.shape[0]
    map_offsets = []
    map_scales = []
    min_gen = 0
    max_gen = 1
    for m in range(N_GOAL_MAPS):
        min_rec = np.min(rec_goal_maps[m])
        max_rec = np.max(rec_goal_maps[m])
        offset = (min_gen * max_rec - min_rec * max_gen) / (max_rec - min_rec)
        scale = (max_gen - min_gen) / (max_rec - min_rec)
        assert scale > 0, "scale parameter should be greater than 0"
        map_offsets.append(offset)
        map_scales.append(scale)
    return map_scales, map_offsets


def transform_recovered_parameters(rec_goal_maps, rec_weights, std_weights=None):
    '''
    perform all transformations for recovered weights and goal maps:
    '''
    #compute scaling factors and offsets to bring goal maps to [0,1]:
    map_scales, map_offsets = compute_scale_and_offset_goal_maps(rec_goal_maps)
    N_GOAL_MAPS = rec_goal_maps.shape[0]
    for k in range(N_GOAL_MAPS):
        rec_goal_maps[k] = map_scales[k]*rec_goal_maps[k].copy() + \
                              map_offsets[k]
        rec_weights[k] = (1 / map_scales[k]) * rec_weights[k].copy()
        if std_weights is not None:
            std_weights[k] = (1 / map_scales[k]) * std_weights[k].copy()

        assert np.isclose(np.min(rec_goal_maps[k]),0,rtol=0.05), "min is not 0"
        assert np.isclose(np.max(rec_goal_maps[k]), 1, rtol=0.001), "max is not 1"
    if std_weights is not None:
        return rec_goal_maps, rec_weights, std_weights 
    else:
        return rec_goal_maps, rec_weights


def get_confidence_intervals(rec_goal_maps, rec_weights, sigma, gamma):
    """ compute the confidence intervals of the recovered weights """
    
    lb = labyrinth_with_stay.LabyrinthEnv()
    P_a = lb.get_transition_mat()
    N_STATES = P_a.shape[0]

    # load expert trajectories
    file = open(TRAJS_DIR_NAME, 'rb')
    expert_trajectories = pickle.load(file)
    N_GOAL_MAPS = rec_goal_maps.shape[0]
    T = len(expert_trajectories[0]["actions"])
    
    # compute inverse hessian of the MAP objective at the recovered parameters
    inv_hess = compute_inv_hessian(seed, P_a, expert_trajectories, [sigma]*N_GOAL_MAPS, rec_weights, rec_goal_maps, gamma)
    # compute confidence intervals now
    std_weights = compute_conf_interval(inv_hess, T, N_GOAL_MAPS, N_STATES)

    return std_weights


def calculate_ll_for_random_policy():
    '''
    a baseline for comparing our model against is a random policy, where each of the actions is selected randomly
    In this case, the loglikelihood of N trajectories with T state-action pairs is N*T*log(1/4)
    '''
    # get length of each trajectory
    file = open(TRAJS_DIR_NAME, 'rb')
    expert_trajectories = pickle.load(file)
    T = len(expert_trajectories[0]["actions"])

    # get number of trajectories 
    val_indices = np.load(GEN_DIR_NAME+"restricted_val_indices.npy").astype(int)
    N = len(val_indices)
    random_ll = np.log(1/4)*N*T
    return random_ll


def get_recovered_params(lamda, gamma, n_map, sigma, lr_map, lr_weight, seed=0, transform=True):
    """ plot recovered goal maps and weights for a given parameter setting """
    
    load_dir = REC_DIR_NAME + 'maps_'+str(n_map)+'_sigma_' +str(sigma)+'_lr_'+str(lr_map)+\
                            '_'+str(lr_weight)+'_'+str(gamma)+'_'+str(lamda)+'/'


    rec_goal_maps = np.load(load_dir + 'goal_maps_trajs_'+str(N_TRAJS)+'_seed_'+str(seed)+'_iters_100.npy')[-1]
    rec_weights = np.load(load_dir + 'weights_trajs_'+str(N_TRAJS)+'_seed_'+str(seed)+'_iters_100.npy')[-1]

    # get confidence intervals
    std_weights = get_confidence_intervals(rec_goal_maps, rec_weights, sigma, gamma)

    # transform the rec maps and weights
    if transform == True:
        final_rec_goal_maps, final_rec_weights, final_std_weights = transform_recovered_parameters(rec_goal_maps,
                                                                                            rec_weights, std_weights)
    else:
        final_rec_goal_maps, final_rec_weights, final_std_weights = rec_goal_maps, rec_weights, std_weights

    rec_rewards = final_rec_weights.T@final_rec_goal_maps
    return final_rec_goal_maps, final_rec_weights, final_std_weights, rec_rewards

            
def get_val_lls_to_plot():
    """ get validation LL for all possible number of maps """
    
    N_MAPS = [1, 2, 3, 4]
    val_lls = []

    for n_maps in N_MAPS:
        load_dir = REC_DIR_NAME + 'maps_' + str(n_maps) + '_sigma_' + str(sigma) + \
                   '_lr_' + str(lr_maps) + '_' + str(lr_weights) + \
                   '_' + str(gamma) + '_' + str(lamda) + '/'
        val_ll_this_setting = np.load(load_dir + 'validation_lls_' + str(N_TRAJS) + \
                                      '_seed_' + str(seed) + '_iters_100.npy')[-1]
        val_lls.append(val_ll_this_setting)

    # get length of trajectory
    file = open(TRAJS_DIR_NAME, 'rb')
    expert_trajectories = pickle.load(file)
    T = len(expert_trajectories[0]["actions"])
    
    num_decisions = (T * 0.2 * N_TRAJS)
    val_lls = np.array(val_lls) / (num_decisions) / np.log(2)
    return val_lls

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='enter training specifics')
    parser.add_argument('--TRAIN_DIRL_NOW', type=int, default=0,
                        help='whether to load and plot previously saved results or train DIRL now')
    args = parser.parse_args()
    TRAIN_DIRL_NOW = args.TRAIN_DIRL_NOW #warning - setting this to True results in the code taking ~20 minutes to run on a laptop

    
    GEN_DIR_NAME = 'data/mouse_data/' # directory that stores trajectories of mice
    TRAJS_DIR_NAME = GEN_DIR_NAME + 'water_restricted_mice_trajs.pickle'
    REC_DIR_NAME = 'recovered_parameters/mice_recovered_params/water_restricted/' # directory that should store parameters recovered by DIRL
    N_TRAJS = 200 # num of trajectories of water-restricted mice
    SAVE_DIR = 'figures/' # directory to save figures in
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # parameters to reproduce Fig 4
    sigma = 0.25 # noise variance of time-varying weights
    lr_maps = 0.005 # learning rate for goal maps
    lr_weights = 0.05 # learning rate for weights
    lamda = 0.001 # l2 coefficient for goal maps
    n_maps = 2 # number of goal maps
    gamma = 0.7 # discount factor in value iteration
    seed = 1 # initialization seed
    max_iters = 100 # max iters to run SGD for optimization of goal maps and weights durng each outer loop of dirl

    if TRAIN_DIRL_NOW:
        # fits DIRL and saves the recovered parameters 
        fit_dirl_mice(True, lr_weights, lr_maps, max_iters, n_maps, sigma, gamma, lamda, seed, GEN_DIR_NAME, REC_DIR_NAME)

    # now obtain recovered parameters post alignment
    final_rec_goal_maps, final_rec_weights, final_std_weights, rec_rewards = \
        get_recovered_params(lamda, gamma, n_maps, sigma, lr_maps, lr_weights, seed=seed, transform=True)

    # ==================== PLOTTING CODE ====================

    STATES_TO_PLOT = [0, 100] # states that we want to plot when showing rewards
    STATE_LABELS = ['home', 'water port'] 
    MAP_LABELS = ["water", "home"] # labels of goal maps

    #======== GOAL MAPS ==========
    fig = plt.figure(figsize=(1.75, 1.5))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.90, top=0.95,
                        wspace=0.1, hspace=0.1)
    # create a maze to plot on
    maze = NewMaze(6)
    # plot the recovered goal maps
    ax = plt.subplot(1, 1, 1)
    converted_map = convert_to_rosenberg_space(final_rec_goal_maps[0])
    PlotMazeFunction(converted_map, maze, mode='nodes', numcol=None, figsize=6, axes=ax,)  # , axes= axs[i//5][i%5])
    plt.title('"water"', fontsize=10)
    plt.axis('off')
    fig.savefig(SAVE_DIR + "restricted_goal_maps_1.pdf")

    # plot the recovered goal maps
    fig = plt.figure(figsize=(1.75, 1.5))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.90, top=0.95,
                        wspace=0.1, hspace=0.1)
    ax = plt.subplot(1, 1, 1)
    converted_map = convert_to_rosenberg_space(final_rec_goal_maps[1])
    PlotMazeFunction(converted_map, maze, mode='nodes', numcol=None, figsize=6, axes=ax,)  # , axes= axs[i//5][i%5])
    plt.title('"home"', fontsize=10)
    plt.axis('off')
    fig.savefig(SAVE_DIR + "restricted_goal_maps_2.pdf")


    # ====== TIME VARYING WEIGHTS =========

    fig = plt.figure(figsize=(2.9, 1))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.90, top=0.95,
                        wspace=0.3, hspace=0.1)

    plt.subplot(1, 2, 1)
    plt.plot(final_rec_weights[0],
             color=colors_list[-1],
             linewidth=1.5)
    plt.fill_between(np.arange(len(final_rec_weights[0])),
                     final_rec_weights[0] - 2 * final_std_weights[0],
                     final_rec_weights[0] + 2 * final_std_weights[0],
                     facecolor='k', alpha=0.2)
    plt.axhline(y=0,
                color='k',
                alpha=0.6,
                linestyle="--",
                linewidth=0.5)
    plt.ylim(-7, 27)
    plt.yticks([0, 25], fontsize=10)
    plt.xticks([0, 20], fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel("weights", fontsize=10)
    plt.xlabel("time", fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)


    plt.subplot(1, 2, 2)
    plt.plot(final_rec_weights[1],
             color=colors_list[-1],
             linewidth=1.5)
    plt.fill_between(np.arange(len(final_rec_weights[1])),
                     final_rec_weights[1] - 2 * final_std_weights[1],
                     final_rec_weights[1] + 2 * final_std_weights[1],
                     facecolor='k', alpha=0.2)
    plt.axhline(y=0,
                color='k',
                alpha=0.6,
                linestyle="--",
                linewidth=0.5)
    plt.yticks([0, 25], ['', ''], fontsize=10)
    plt.xticks([0, 20], ['', ''], fontsize=10)
    plt.ylim(-7, 27)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()
    fig.savefig(SAVE_DIR + "restricted_time_varying_weights.pdf")


    # === REWARDS ===

    # plot the recovered time-varying rewards
    fig = plt.figure(figsize=(2, 1))
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.90, top=0.95,
                        wspace=0.3, hspace=0.1)
    plt.subplot(1, 1, 1)
    plt.plot(rec_rewards[:, 0],
             color=colors_list[1],
             linewidth=1.5, label=STATE_LABELS[0])
    plt.plot(rec_rewards[:, 100],
             color=colors_list[0], #water should be blue! switch colors
             linewidth=1.5, label=STATE_LABELS[1])
    # let's now plot the avg for the other states:
    avg_rewards = np.zeros(rec_rewards.shape[0])
    counter = 0
    for state in range(rec_rewards.shape[1]):
        if state not in STATES_TO_PLOT:
            avg_rewards += rec_rewards[:, state]
            counter += 1
    avg_rewards = (1 / counter) * avg_rewards
    plt.plot(avg_rewards,
             color=colors_list[2], linewidth=1.5,  zorder=2, )
    plt.title("rewards")
    plt.axhline(y=0,
                color='k',
                alpha=0.6,
                linestyle="--",
                linewidth=0.5)
    plt.ylim(-7, 25)
    plt.yticks([0, 25], fontsize=10)
    plt.xticks([0, 20], fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()
    fig.savefig(SAVE_DIR + "restricted_time_varying_rewards.pdf")

    #======== XVAL PLOT =============
    cols=['#999999', '#984ea3', '#e41a1c', '#dede00']

    if TRAIN_DIRL_NOW:
        # train 3 more models with different n_maps to make this plot
        possible_n_maps = [1,2,3,4]
        possible_n_maps.remove(n_maps)
        for n_maps in possible_n_maps:
            # fits DIRL and saves the recovered parameters 
            fit_dirl_mice(True, lr_weights, lr_maps, max_iters, n_maps, sigma, gamma, lamda, seed, GEN_DIR_NAME, REC_DIR_NAME)

    # obtain val lls for DIRL at varying number of maps
    val_lls_dirl = get_val_lls_to_plot()
    # obtain val LL for a random policy
    random_val_ll = calculate_ll_for_random_policy()
    # load val LL when using MaxEnt 
    max_ent_val_ll = np.load('recovered_parameters/existing_irl_methods_results/maxent_best_ll_restricted.npy')
    # load val LL when using DeepMaxEnt
    deep_max_ent_val_ll = np.load('recovered_parameters/existing_irl_methods_results/deepmaxent_best_ll_restricted.npy')

    vals_to_plot = np.concatenate(([random_val_ll], [max_ent_val_ll], [deep_max_ent_val_ll], val_lls_dirl))

    fig, ax = plt.subplots(figsize=(2, 1.5))
    plt.subplots_adjust(left=0.05, bottom=0.25, right=0.9, top=0.8)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    for z, ba in enumerate(vals_to_plot):
        if z <=2 :
            col = cols[1]
        else:
            col = cols[2]
        plt.barh( 12-(2*z), ba, color=col, height=1.4)
    plt.yticks([0, 2, 4, 6, 8, 10, 12], ['', '', '', '', '', '', ''], fontsize=10)
    plt.xlim((-0.5, -2.02))
    plt.xticks([-2, -1.5, -1.0, -0.5], ['-2', '-1.5', '-1', '-0.5'], fontsize=10)
    plt.gca().invert_xaxis()
    plt.title("test LL", fontsize=10)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(SAVE_DIR + "restricted_model_comparison.pdf")
    plt.show()