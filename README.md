# dynamic_irl
Code to reproduce Figures 3, 4 and 5 of "Dynamic Inverse Reinforcement Learning for Characterizing Animal Behavior".  Note: we start with Figure 3, as this is the first figure to apply DIRL to data. The first two figures are the DIRL model schematic and a figure showing the raw trajectory data.  

To get started, first create a conda environment from the environment yaml file:
```
conda env create -f environment.yml
```

Activate this environment by running:
```
conda activate dirl
```

### Figure 3

In order to reproduce Figure 3 (application of DIRL to simulated trajectories in the gridworld environment), run:
```
python make_figure_3.py
```
This plots the recovered parameters (saved in `recovered_parameters/gridworld_recovered_params`) obtained by fitting DIRL on simulated trajectories in the 5x5 gridworld (the simulated trajectories are saved in `data/simulated_gridworld_data`). If you want to refit DIRL to the trajectory data, run the following:
```
python make_figure_3.py --TRAIN_DIRL_NOW=1
```

Finally, if you want to simulate a new set of trajectories in the 5x5 gridworld, this can be done by running the code in `src/simulate_data_gridworld.py`.

### Figure 4

In order to reproduce Figure 4 (application of DIRL to the trajectories of water-restricted mice from 
et al., 2021), run:
```
python make_figure_4.py
```
This plots the recovered parameters (saved in `recovered_parameters/mice_recovered_params`) obtained by fitting DIRL on the trajectories of water-restricted mice in the labyrinth of Rosenberg et al., 2021 (the trajectories are saved in `data/mouse_data`). If you want to refit DIRL to the trajectory data, run the following:
```
python make_figure_4.py --TRAIN_DIRL_NOW=1
```

### Figure 5

In order to reproduce Figure 5 (application of DIRL to the trajectories of water-unrestricted mice from Rosenberg et al., 2021), run:
```
python make_figure_5.py
```
This plots the recovered parameters (saved in `recovered_parameters/mice_recovered_params`) obtained by fitting DIRL on the trajectories of water-unrestricted mice in the labyrinth of Rosenberg et al., 2021 (the trajectories are saved in `data/mouse_data`). If you want to refit DIRL to the trajectory data, run the following:
```
python make_figure_5.py --TRAIN_DIRL_NOW=1
```

Warning: fitting DIRL can take up to 10 minutes to run on a laptop.  

Note: the scripts `MM_Maze_Utils.py` and `MM_Plot_Utils.py` in the `plot_utils` folder are copies of scripts from the original code base associated with the Rosenberg et al. paper. We use these here for plotting purposes.

Finally, the scripts above produce figures that are saved in the `figures/` directory.



