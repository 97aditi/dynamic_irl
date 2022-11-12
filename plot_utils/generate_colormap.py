from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def generate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ truncates a colormap 
        args:
            cmap (object): a colormap
            min_val (float): min val at which to truncate
            max_val (float): max val at which to truncate
            n (int): number of divisions of the colormap to decie truncation points 
        returns:
            new_cmap (object): new truncated cmap """
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap