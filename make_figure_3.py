import numpy as np
import matplotlib.pyplot as plt
import argparse, os
import pickle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from itertools import permutations
from src.compute_conf_interval import compute_conf_interval, compute_inv_hessian
from src.dirl_for_gridworld import fit_dirl_gridworld
from plot_utils.generate_colormap import generate_colormap

# MATPLOTLIB settings
LEGEND_SIZE = 10
SMALL_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rcParams['font.sans-serif'] = "Helvetica"
colors = ['steelblue', '#D85427', 'tab:green', 'k']


def compute_scale_and_offset_goal_maps(gen_goal_maps, rec_goal_maps):
    '''
    scale recovered goal maps so that generative and recovered goal
    maps have the same maximum and minimum values
    '''
    N_goal_MAPS = gen_goal_maps.shape[0]
    map_offsets = []
    map_scales = []
    for m in range(N_goal_MAPS):
        min_gen = np.min(gen_goal_maps[m])
        max_gen = np.max(gen_goal_maps[m])
        min_rec = np.min(rec_goal_maps[m])
        max_rec = np.max(rec_goal_maps[m])
        offset = (min_gen * max_rec - min_rec * max_gen) / (max_rec - min_rec)
        scale = (max_gen - min_gen) / (max_rec - min_rec)
        assert scale > 0, "scale parameter should be greater than 0"
        map_offsets.append(offset)
        map_scales.append(scale)
    return map_scales, map_offsets


def transform_recovered_parameters(gen_goal_maps, gen_weights,
                                   rec_goal_maps, rec_weights, std_weights=None, std_maps=None):
    '''
    perform all transformations for recovered weights and goal maps:
    perform the sign conversion and calculate the relevant scales and offsets
    '''
    #calculate the sign conversion between recovered and generative
    # parameters.  Use time-varying weights to calculate sign since these
    # are only going to be modified by a positive scaling factor later on
    N_GOAL_MAPS = len(gen_goal_maps)
    for k in range(N_GOAL_MAPS):
        # modifying sign conversion 
        diff_orig = np.linalg.norm(rec_weights[k]-gen_weights[k])
        diff_flipped = np.linalg.norm(rec_weights[k]+gen_weights[k])
        if diff_orig>diff_flipped:
            rec_weights[k] = -1 * rec_weights[k].copy()
            rec_goal_maps[k] = -1 * rec_goal_maps[k].copy()

    #now compute scaling factors and offsets:
    map_scales, map_offsets = compute_scale_and_offset_goal_maps(
        gen_goal_maps, rec_goal_maps)

    for k in range(N_GOAL_MAPS):
        rec_goal_maps[k] = map_scales[k]*rec_goal_maps[k].copy() + \
                              map_offsets[k]
        rec_weights[k] = (1 / map_scales[k]) * rec_weights[k].copy()
        if std_weights is not None:
            std_weights[k] = (1 / map_scales[k]) * std_weights[k].copy()
        if std_maps is not None:
            std_maps[k] = map_scales[k]*std_maps[k].copy() 
    if std_weights is None:
        return rec_goal_maps, rec_weights
    else:
        return rec_goal_maps, rec_weights, std_weights


def calculate_permutation(gen_goal_maps, gen_weights, rec_goal_maps,
                          rec_weights):
    '''
    loop through all permutations, perform appropriate transformations and
    calculate the distance between the generative and recovered parameters.
    identify the permutation of the recovered weights and goal maps so as
    to minimize the distance between generative and recovered parameters
    '''
    N_GOAL_MAPS = gen_goal_maps.shape[0]
    perms = list(permutations(range(N_GOAL_MAPS)))
    dist_vec = []
    for permutation in perms:
        permuted_maps = rec_goal_maps[np.array(permutation)]
        permuted_weights = rec_weights[np.array(permutation)]
        # calculate transformation:
        final_permuted_goal_maps, final_permuted_weights = \
            transform_recovered_parameters(
            gen_goal_maps, gen_weights, permuted_maps, permuted_weights)
        dist_vec.append(
            np.linalg.norm(final_permuted_goal_maps - gen_goal_maps))
    optimal_permutation = perms[np.argmin(dist_vec)]
    return optimal_permutation


def get_confidence_intervals(rec_goal_maps, rec_weights):
    """ compute the confidence intervals of the recovered weights
        returns:
            std_weights (TXN_MAPS): std dev of weights """
    
    # load parameters
    P_a = pickle.load(open(GEN_DIR_NAME + 
                                    "/generative_parameters.pickle", 'rb'))['P_a']
    sigma = pickle.load(open(GEN_DIR_NAME + 
                                    "/generative_parameters.pickle", 'rb'))['sigmas'][0]
    # load expert trajectories
    all_trajectories = pickle.load(open(GEN_DIR_NAME + "/expert_trajectories.pickle", 'rb'))
    val_indices = np.arange(start=0, stop=num_trajs, step=5)
    train_indices = np.delete(np.arange(num_trajs), val_indices)
    expert_trajectories = [all_trajectories[train_idx] for train_idx in train_indices]

    N_GOAL_MAPS = rec_goal_maps.shape[0]
    N_STATES = grid_H*grid_W
    T = len(expert_trajectories[0]["actions"])

    # compute inverse hessian of the MAP objective at the recovered parameters
    inv_hess = compute_inv_hessian(seed, P_a, expert_trajectories, [sigma]*N_GOAL_MAPS, rec_weights, rec_goal_maps, gamma=0.9)
    # compute confidence intervals now
    std_weights = compute_conf_interval(inv_hess, T, N_GOAL_MAPS, N_STATES)
    return std_weights




def get_gen_recovered_parameters(seed, n_maps, lr_weights, lr_maps):
    """ makes a summary plot for recovered and generative parameters
        args:
            seed (int): which seed to plot
            n_maps (int): how many goal maps to pot
            lr_weights (float): which learning rate to plot
            lr_maps (float): which learning rate to plot
            save (bool): whether to save the plot or not
    """
    # directory for loading recovered and generative weights

    rec_dir_name = REC_DIR_NAME + "/maps_"+str(n_maps)+\
                            "_lr_"+str(lr_maps)+"_"+ str(lr_weights) + "/"

    # load generative weights (this is stored as T+1 x N_MAPS)
    gen_time_varying_weights = pickle.load(open(GEN_DIR_NAME + 
                                    "/generative_parameters.pickle", 'rb'))['time_varying_weights']
    # convert this to N_MAPS X T to make it consistent with recovered weights that we will load later
    gen_time_varying_weights = gen_time_varying_weights[:-1].T
    # load goal maps
    gen_goal_maps = pickle.load(open(GEN_DIR_NAME +
                                     "/generative_parameters.pickle", 'rb'))['goal_maps'] 
    # compute generative rewards
    gen_rewards = gen_time_varying_weights.T @ gen_goal_maps
    # recovered weights for the explore state
    min_gen_rewards = np.min(gen_rewards, axis=1)

    # no of goal maps, time steps
    N_GOAL_MAPS, T = gen_time_varying_weights.shape[0],gen_time_varying_weights.shape[1]
    assert N_GOAL_MAPS == n_maps, "we've got a problem with goal map recovery!"

    # load recovered parameters for this seed
    rec_weights = np.load(rec_dir_name + "weights_trajs_" + str(num_trajs) +
                            "_seed_" + str(seed) + "_iters_" + str(max_iters) +
                            ".npy")[-1]
    # load recovered parameters for this seed
    rec_goal_maps = np.load(rec_dir_name + "goal_maps_trajs_" +
                                str(num_trajs) + "_seed_" + str(seed) +
                                "_iters_" + str(max_iters) + ".npy")[-1]

    # compute rewards
    rec_rewards = rec_weights.T @ rec_goal_maps

    # get confidence intervals for weights
    std_weights = get_confidence_intervals(rec_goal_maps, rec_weights)

    # offset for the recovered rewards
    min_rec_rewards = np.min(rec_rewards, axis=1)
    offset = (min_rec_rewards - min_gen_rewards)[:, np.newaxis]

    assert offset.shape[0]==T, "The offset should be computed per time " \
                                "point, and should be the same across all states"

    permutation = calculate_permutation(gen_goal_maps,
                                        gen_time_varying_weights,
                                        rec_goal_maps, rec_weights)

    final_rec_goal_maps, final_rec_weights, final_std_weights = transform_recovered_parameters(gen_goal_maps,
                                    gen_time_varying_weights,
                                    rec_goal_maps[
                                        np.array(permutation)],
                                    rec_weights[np.array(permutation)], std_weights[np.array(permutation)],)

    rec_rewards = rec_rewards - offset


    return gen_goal_maps, gen_time_varying_weights, final_rec_goal_maps, final_rec_weights,final_std_weights, gen_rewards, rec_rewards



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='enter environment specifics')
    parser.add_argument('--TRAIN_DIRL_NOW', type=int, default=0,
                        help='whether to load and plot previously saved results or train DIRL now')
    args = parser.parse_args()
    TRAIN_DIRL_NOW = args.TRAIN_DIRL_NOW #warning - setting this to True results in the code taking ~20 minutes to run on a laptop

    VERSION = 1 # version of the generative trajectories (these trajectories are generated using simulate_data_gridworld.py)

    # name of directory that contains simulated trajectories
    GEN_DIR_NAME = 'data/simulated_gridworld_data'
    # name of directory that should store the recovered parameters
    REC_DIR_NAME = 'recovered_parameters/gridworld_recovered_params'
    REC_DIR_NAME = REC_DIR_NAME + '/exclude_explore'+ "_"+str(VERSION)
    GEN_DIR_NAME = GEN_DIR_NAME + "/exclude_explore_share_weights"+ "_"+str(VERSION)
    SAVE_DIR = 'figures/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    grid_H, grid_W = 5, 5 # size of gridworld
    
    num_trajs = 200 # number of simulated trajectories to use
    # parameters to reproduce Fig 3
    max_iters = 100 # max iters to run SGD for optimization of goal maps and weights durng each outer loop of dirl
    n_maps = 2 # goal maps
    lr_maps = 0.001 # lr of goal maps
    lr_weights = 0.05 # lr of weights
    seed = 1 # initialization seed
    gamma = 0.9 # discount factor

    if TRAIN_DIRL_NOW:
        # fits DIRL and saves the recovered parameters 
        fit_dirl_gridworld(num_trajs, lr_weights, lr_maps, max_iters, gamma, n_maps, seed, GEN_DIR_NAME, REC_DIR_NAME)
    
    # now obtain generative and recovered (post alignment) parameters
    gen_goal_maps, gen_time_varying_weights, final_rec_goal_maps, final_rec_weights, final_std_weights, gen_rewards, rec_rewards = \
        get_gen_recovered_parameters(seed, n_maps, lr_weights, lr_maps)

    # =================================
    # ========= BEGIN PLOT ============
    
    MAP_LABELS = ['home', 'water'] # labels of the goal maps
    STATE_LABELS = ['home', 'water'] # labels of states in gridworld to plot rewards for
    STATES_TO_PLOT = [0, 14]


    # EXAMPLE TRAJECTORIES
    traj_indices = [10, 40]
    fig, axs = plt.subplots(2, 1, figsize=(1.1, 2.2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.3, hspace=0.4)

    plt.subplot(2, 1, 1)
    traj = pickle.load(open(GEN_DIR_NAME + "/expert_trajectories.pickle", 'rb'))[traj_indices[0]]
    plt.imshow(np.zeros((grid_H, grid_W)), alpha=0)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[])
    plt.xticks([])
    plt.yticks([])
    # set up a list of segments
    states2d = traj['states2d']
    x = states2d[:, 1] + np.random.normal(scale=0.1, size=len(states2d[:, 1]))  # jitter traj slightly
    y = states2d[:, 0] + np.random.normal(scale=0.1, size=len(states2d[:, 0]))  # jitter traj slightly
    t = np.linspace(0, 1, len(x))

    points = np.array([x, y]).transpose().reshape(-1, 1, 2)

    # set up a list of segments
    segs = np.concatenate([points[:-1], points[1:]], axis=1)  # connect current state to next state

    # make the collection of segments
    lc = LineCollection(segs, cmap=plt.get_cmap('plasma'), linewidths=1)
    lc.set_array(t)  # color the segments by our parameter
    lines = axs[0].add_collection(lc);  # add the collection to the plot
    cbar = plt.colorbar(lines, fraction=0.046, pad=0.04, location='left')
    cbar.set_ticks([0, 1])

    plt.subplot(2, 1, 2)
    traj = pickle.load(open(GEN_DIR_NAME + "/expert_trajectories.pickle", 'rb'))[traj_indices[1]]
    plt.imshow(np.zeros((grid_H, grid_W)), alpha=0)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[])
    plt.xticks([])
    plt.yticks([])
    # set up a list of segments
    states2d = traj['states2d']
    x = states2d[:, 1] + np.random.normal(scale=0.1, size=len(states2d[:, 1]))  # jitter traj slightly
    y = states2d[:, 0] + np.random.normal(scale=0.1, size=len(states2d[:, 0]))  # jitter traj slightly
    t = np.linspace(0, 1, len(x))

    points = np.array([x, y]).transpose().reshape(-1, 1, 2)

    # set up a list of segments
    segs = np.concatenate([points[:-1], points[1:]], axis=1)  # connect current state to next state

    # make the collection of segments
    lc = LineCollection(segs, cmap=plt.get_cmap('plasma'), linewidths=1)  # jet, viridis hot
    lc.set_array(t)  # color the segments by our parameter
    lines = axs[1].add_collection(lc);  # add the collection to the plot
    cbar = plt.colorbar(lines, fraction=0.046, pad=0.04, location='left')
    cbar.set_ticks([0, 1])
    plt.show()
    fig.savefig(SAVE_DIR + 'figure_3_trajs.pdf')

    fig = plt.figure(figsize=(2.2, 2.2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93,
                        wspace=0.3, hspace=0.4)

    # WATER MAP
    plt.subplot(2, 2, 1)
    cmap = plt.get_cmap('viridis')
    new_cmap = generate_colormap(cmap, 0.5, 1.0)

    plt.imshow(np.reshape(gen_goal_maps[0],
                          (grid_H, grid_W),
                          order='F'), vmin=0, vmax=1, cmap=new_cmap)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[])
    plt.title('gen. ' + MAP_LABELS[0], fontsize=8)
    plt.axis('off')

    #HOME MAP
    plt.subplot(2, 2, 2)
    cmap = plt.get_cmap('viridis')
    new_cmap = generate_colormap(cmap, 0.5, 1.0)

    plt.imshow(np.reshape(gen_goal_maps[1],
                          (grid_H, grid_W),
                          order='F'), vmin=0, vmax=1, cmap=new_cmap)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[])
    plt.title('gen. ' + MAP_LABELS[1], fontsize=8)
    plt.axis('off')


    # RECOVERED WATER MAP
    plt.subplot(2, 2, 3)
    plt.imshow(np.reshape(final_rec_goal_maps[0],
                          (grid_H, grid_W),
                          order='F'), vmin=0, vmax=1, cmap=new_cmap)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[])
    plt.title('rec. ' + MAP_LABELS[0], fontsize=8)
    plt.axis('off')

    # RECOVERED HOME MAP
    plt.subplot(2, 2, 4)
    plt.imshow(np.reshape(final_rec_goal_maps[1],
                          (grid_H, grid_W),
                          order='F'), vmin=0, vmax=1, cmap=new_cmap)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[0, 1])
    plt.title('rec. ' + MAP_LABELS[1], fontsize=8)
    plt.axis('off')
    plt.show()
    fig.savefig(SAVE_DIR + 'figure_3_maps.pdf')


    # plot the recovered weights
    fig = plt.figure(figsize=(2.5, 0.95))
    plt.subplots_adjust(left=0.1, bottom=0.23, right=0.95, top=0.95,
                        wspace=0.4, hspace=0.5)

    plt.subplot(1, 2, 1)
    plt.plot(final_rec_weights[0], color=colors[-1], linewidth=1.5,
             linestyle="--", zorder=2, label="rec.")
    plt.fill_between(np.arange(len(final_rec_weights[0])),
                     final_rec_weights[0] - 2 * final_std_weights[0],
                     final_rec_weights[0] + 2 * final_std_weights[0],
                     facecolor='k', alpha=0.1)
    plt.plot(gen_time_varying_weights[0], label='gen.',
             color=colors[-1], linewidth=2, zorder=1, alpha=0.5)
    plt.axhline(y=0, color='k', alpha=0.6, linestyle="--", linewidth=0.5)
    plt.ylim(-1, 2)
    plt.yticks([0, 2], fontsize=10)
    plt.xticks(fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # WEIGHTS 2
    plt.subplot(1, 2, 2)
    plt.plot(gen_time_varying_weights[1], label='gen.',
             color=colors[-1], linewidth=2, zorder=1, alpha=0.5)
    plt.plot(final_rec_weights[1], color=colors[-1], linewidth=1.5,
             linestyle="--", zorder=2, label="rec.")
    plt.fill_between(np.arange(len(final_rec_weights[1])),
                     final_rec_weights[1] - 2 * final_std_weights[0],
                     final_rec_weights[1] + 2 * final_std_weights[0],
                     facecolor='k', alpha=0.1)
    plt.axhline(y=0, color='k', alpha=0.6, linestyle="--", linewidth=0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.yticks([0, 2], ['', ''])
    plt.xticks(fontsize=10)
    plt.ylim(-1, 2)
    plt.show()
    fig.savefig(SAVE_DIR + "figure_3_weights.pdf")


    # plot the recovered rewards
    fig = plt.figure(figsize=(2.5, 0.95))
    plt.subplots_adjust(left=0.1, bottom=0.23, right=0.95, top=0.95,
                        wspace=0.4, hspace=0.5)
    plt.subplot(1, 2, 1)
    for num_state in range(len(STATE_LABELS)):
        plt.plot(rec_rewards[:, STATES_TO_PLOT[num_state]], color=colors[num_state],
                 linewidth=1.5, linestyle="--", zorder=2, label=STATE_LABELS[num_state])
        plt.plot(gen_rewards[:, STATES_TO_PLOT[num_state]],
                 color=colors[num_state], linewidth=2, zorder=1, alpha=0.5)
    # let's now plot the avg for the other states:
    avg_rewards = np.zeros(rec_rewards.shape[0])
    counter = 0
    for state in range(rec_rewards.shape[1]):
        if state not in STATES_TO_PLOT:
            avg_rewards += rec_rewards[:, state]
            counter+=1
    avg_rewards = (1/counter)*avg_rewards
    plt.plot(avg_rewards,
             color=colors[2], linewidth=1.5, linestyle="--", zorder=2,label='avg other states' )
    plt.plot(np.zeros(rec_rewards.shape[0]),
             color=colors[2], linewidth=2, zorder=1, alpha=0.5)
    plt.axhline(y=0, color='k', alpha=0.6, linestyle="--", linewidth=0.5)
    plt.ylim(-1, 2)
    plt.yticks([0, 2], fontsize=10)
    plt.xticks([0, 50], fontsize=10)
    plt.title("rewards", fontsize=10)
    plt.xlabel("time", fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(fontsize=10)

    # plot val_lls across the 4 different maps
    plt.subplot(1, 2, 2)

    if TRAIN_DIRL_NOW:
        # train 3 more models with different n_maps to make this plot
        possible_n_maps = [1,2,3,4]
        possible_n_maps.remove(n_maps)
        for n_maps in possible_n_maps:
            fit_dirl_gridworld(num_trajs, lr_weights, lr_maps, max_iters, gamma, n_maps, seed, GEN_DIR_NAME, REC_DIR_NAME)

    # load validation LLs to plot
    val_lls_to_plot = []
    for n_maps in [1,2,3,4]:
        load_folder = REC_DIR_NAME + "/maps_"+str(n_maps)+\
                            "_lr_"+str(lr_maps)+"_"+ str(lr_weights) + "/"
        val_ll = np.load(load_folder + '/validation_lls_'+str(num_trajs)+"_seed_"+str(seed)+\
                            "_iters_"+str(max_iters)+".npy")[-1]
        val_lls_to_plot.append(val_ll)
    val_lls_to_plot = np.array(val_lls_to_plot)
    plt.plot(np.arange(len(val_lls_to_plot)) + 1, val_lls_to_plot / (40 * 50) / np.log(2), color='k', marker='o',
             linewidth=1, markersize=2)
    plt.xlabel("# of maps", fontsize=10)
    plt.title("test LL \n(bits/decision)", fontsize=10)
    plt.yticks([-2.15, -2.1],['-2.15', '-2.1'], fontsize=10)
    plt.xticks([1,2,3,4], fontsize=10)
    plt.ylim((-2.15, -2.08))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()
    fig.savefig(SAVE_DIR + "figure_3_rewards.pdf")