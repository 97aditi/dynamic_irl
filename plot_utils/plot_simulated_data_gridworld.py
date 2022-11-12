# Make plots of reward function, as well as trajectories in maze
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

def plot_gridworld_trajectories(grid_H, grid_W, traj, ax):
    '''
    plot trajectory in gridworld.  use color gradient to indicate time
    '''

    plt.imshow(np.zeros((grid_H,grid_W)), alpha=0)

    counter = 0
    for x in range(grid_H):
        for y in range(grid_W):
            plt.text(x, y, counter,
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
            counter+=1

    # set up a list of segments
    states2d = traj['states2d']
    x = states2d[:, 1] + np.random.normal(scale=0.1, size=len(states2d[:, 1]))  # jitter traj slightly
    y = states2d[:,0] + np.random.normal(scale=0.1, size=len(states2d[:,0])) #jitter traj slightly
    t = np.linspace(0,1,len(x))

    points = np.array([x, y]).transpose().reshape(-1, 1, 2)

    # set up a list of segments
    segs = np.concatenate([points[:-1],points[1:]],axis=1) #connect current state to next state

    # make the collection of segments
    lc = LineCollection(segs, cmap=plt.get_cmap('viridis'), linewidths=2)
    lc.set_array(t)  # color the segments by our parameter
    lines = ax.add_collection(lc);  # add the collection to the plot
    cbar = plt.colorbar(lines)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Start', 'End'])


def plot_rewards_all(goal_maps, time_varying_weights, thetas, grid_H, grid_W,LOCATION_WATER, LOCATION_HOME,
                              save_name):
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.3, hspace=0.3)
    gs = fig.add_gridspec(3, len(goal_maps))
    # goal maps
    labels = ['home', 'water', 'explore']
    for m in range(1, len(goal_maps)+1):
        plt.subplot(3, len(goal_maps), m)
        plt.imshow(np.reshape(goal_maps[m-1], (grid_H, grid_W), order='F'), vmin=0,
                   vmax=1)
        counter = 0
        for x in range(grid_H):
            for y in range(grid_W):
                plt.text(x, y, counter,
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
                counter += 1
        plt.title('goal map ' + str(m) + '; ' + labels[m-1])
        plt.axis('off')

    # plot time-varying weights
    for m in range(1, len(goal_maps) + 1):
        plt.subplot(3, len(goal_maps), m+len(goal_maps))
        plt.plot(time_varying_weights[:, m-1])
        plt.xlabel("time")
        plt.ylabel("weights (" + str(m) + ")")
        plt.ylim((-1.5, 2))
        plt.axhline(y=0, color='k', alpha = 0.5)

    ax = fig.add_subplot(gs[2, :])
    plt.plot(thetas[LOCATION_HOME, :], label = 'home')
    plt.plot(thetas[LOCATION_WATER,:], label='water')
    plt.plot(thetas[1, :], label='other states')
    plt.xlabel("time")
    plt.ylabel("reward")
    plt.ylim((-1.5, 2))
    plt.legend()
    fig.savefig(save_name)





