import numpy as np
import itertools
import matplotlib.pyplot as plt
from functions import torch2np_u8

def rgbd_gradients_dataset_first_n(dataset, n, random_start=True, **kw):
    """
    Plots first n images of a dataset containing tensor images.
    """
    if random_start:
        start = np.random.randint(0, len(dataset) - n)
        stop  = start + n
    else:
        start = 0
        stop  = n

    # [(img0, cls0), ..., # (imgN, clsN)]
    first_n = list(itertools.islice(dataset, start, stop))
    return rgbd_gradients_dataset_plot(first_n, **kw)

def rgbd_gradients_dataset_plot(samples: list, figsize=(12, 12), wspace=0.1, hspace=0.2, cmap=None):
    nrows = len(samples)
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             gridspec_kw=dict(wspace=wspace, hspace=hspace),
                             subplot_kw={'aspect': 1})
    # Plot each tensor
    for i in range(len(samples)):
        rgb   = samples[i]['rgb']
        depth = samples[i]['depth']
        x     = samples[i]['x'].squeeze(0)
        y     = samples[i]['y'].squeeze(0)

        rgb   = torch2np_u8(rgb)
        depth = torch2np_u8(depth)

        axes[i,0].imshow(rgb,   cmap=cmap)
        axes[i,1].imshow(depth, cmap=cmap)

        X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
        U,V = x,y
        axes[i,2].quiver(X, Y, U, V, pivot='tip', units='xy')

    return fig, axes