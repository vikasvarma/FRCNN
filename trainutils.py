import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

#-------------------------------------------------------------------------------
def batchplot(batch):
    batch_size = len(batch[0])
    n_ax = int(math.sqrt(batch_size))
    fig, ax = plt.subplots(n_ax, n_ax)
    ax = ax.flatten()
    
    patches   = batch[0]
    batch_roi = batch[1]
        
    for n in range(batch_size):
        # Display the image
        image = patches[n].permute(1,2,0).numpy()
        ax[n].imshow(image)
            
        rois = batch_roi[batch_roi[:,0] == n,:]
        for p in range(rois.size(0)):
                
            roi_origin = rois[p,[2, 1]].int()
            h = rois[p,3] - rois[p,1] + 1
            w = rois[p,4] - rois[p,2] + 1

            rect = Rectangle(roi_origin.tolist(), int(w), int(h),
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')

            # Add the patch to the Axes
            ax[n].add_patch(rect)

    plt.show()