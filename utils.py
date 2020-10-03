import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from matplotlib.patches import Rectangle

#-------------------------------------------------------------------------------
def batchplot(batch, groundTruth, pred, labels):
    batch_size = batch.size(0)
    n_ax = int(math.sqrt(batch_size))
    fig, ax = plt.subplots(n_ax, n_ax)
    ax = ax.flatten()
    
    if pred.dim() == 3:
        # Flatten batch ROIs:
        batch_num = torch.from_numpy(np.repeat(range(batch_size), pred.size(1)))
        batch_num = batch_num.to(pred.dtype)
        pred = torch.cat((batch_num.view(-1,1), pred.view(-1,4)), dim=1)
        
    for n in range(batch_size):
        # Display the image
        image = batch[n].permute(1,2,0).detach().numpy()
        image = (image + 1) / 2
        ax[n].imshow(image)
            
        rois = groundTruth[n]
        roi_origin = rois[:,[1, 0]].int()
        h = rois[:,2] - rois[:,0] + 1
        w = rois[:,3] - rois[:,1] + 1

        valid = (w > 1) & (h > 1)
        roi_origin = roi_origin[valid,:]
        h = h[valid]
        w = w[valid]
            
        for p in range(roi_origin.size(0)):
            rect = Rectangle(roi_origin[p].tolist(), int(w[p]), int(h[p]),
                             linewidth=2,
                             edgecolor='b',
                             facecolor='none')

            # Add the patch to the Axes
            ax[n].add_patch(rect)

        # Select only positive ROI samples:
        positive = labels[n,:].eq(1)
        rois = pred[pred[:,0] == n,:]
        rois = rois[positive, :]
        
        # Only keep rois with positive sizes:
        keep = ((rois[:,3] - rois[:,1] > 0) & (rois[:,4] - rois[:,2] > 0))
        rois = rois[keep,:]
            
        for p in range(rois.size(0)):
                
            roi_origin = rois[p,[2, 1]].int()
            h = rois[p,3] - rois[p,1] + 1
            w = rois[p,4] - rois[p,2] + 1

            rect = Rectangle(roi_origin.tolist(), int(w), int(h),
                            linewidth=0.5,
                            edgecolor='r',
                            facecolor='none')

            # Add the patch to the Axes
            ax[n].add_patch(rect)

    plt.show()
    
#-------------------------------------------------------------------------------
class SizeEstimator(object):

    def __init__(self, model, input_size=(1,1,32,32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = 32
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            sizes = []
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        with torch.no_grad():
            input_ = Variable(torch.FloatTensor(*self.input_size))
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size))*self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total