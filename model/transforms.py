import numpy as np
import torch
from torchvision.transforms import ColorJitter, Resize, Normalize
from PIL import Image

# ------------------------------------------------------------------------------
#   Image Resize
# ------------------------------------------------------------------------------
class ImageResize(object):
    """Resize image and targets to specified size.
    """

    # --------------------------------------------------------------------------
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
        self.tform = Resize(self.output_size)

    # --------------------------------------------------------------------------
    def __call__(self, inputs):
        """
        Random contextual image crop to desired image size.
        """
        image = inputs['image']
        H, W  = image.shape[:2]
        rois  = inputs['ROI']
        image = self.tform.__call__(Image.fromarray(image))
        image = np.array(image)
        
        # Resize ROIs:
        scale = self.output_size / np.array([H,W], dtype=np.float)
        scale = np.tile(scale, 2)
        rois[:,:4] = scale * rois[:,:4]
        rois = np.round(rois)

        return {'image': image, 'ROI': rois}
    
# ------------------------------------------------------------------------------
#   Randomized Image Crop
# ------------------------------------------------------------------------------
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        border_size (tuple or int): Desired crop border size. If int, square crop is made.
    """

    # --------------------------------------------------------------------------
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    # --------------------------------------------------------------------------
    def __call__(self, inputs):
        """
        Random contextual image crop to desired image size.
        """
        image = inputs['image']
        rois  = inputs['ROI']
        H, W  = image.shape[:2]
        
        # Contextual aware crop: Crop to maximally retain ROIs.
        #   Identify the region containing the ROIs and crop outside this 
        #   region.
        roi_top  = np.amin(rois[:,0:2], axis=0)
        roi_left = np.amax(rois[:,2:4], axis=0)
        
        region = np.zeros(4, dtype=int)
        
        # Sample row indices:
        if H - roi_left[0] < roi_top[0] + 1:
            # ROIs closer to the lower edge. Sample the y2 region location.
            regend    = min(max(roi_left[0], self.output_size[0])+1, H-1)
            region[2] = np.random.randint(regend, H)
            region[0] = region[2] - self.output_size[0] + 1
        else:
            # ROIs closer to the upper edge. Sample the y1 region location.
            regstart  = max(min(roi_top[0], H-self.output_size[0]), 1)
            region[0] = np.random.randint(0, regstart)
            region[2] = region[0] + self.output_size[0] - 1
            
        # Sample col indices:
        if W - roi_left[1] < roi_top[1] + 1:
            # ROIs closer to the right edge. Sample the x2 region location.
            regend    = min(max(roi_left[1], self.output_size[1])+1, W-1)
            region[3] = np.random.randint(regend, W)
            region[1] = region[3] - self.output_size[1] + 1
        else:
            # ROIs closer to the left edge. Sample the x1 region location.
            regstart  = max(min(roi_top[1], W-self.output_size[1]), 1)
            region[1] = np.random.randint(0, regstart)
            region[3] = region[1] + self.output_size[1] - 1
            
        # Crop image at sampled region:
        image = image[region[0]:region[2]+1, region[1]:region[3]+1]

        # Adjust ROI locations to image origin shift and only retain the ones 
        # that fit inside the cropped region.
        rois[:,:4] = rois[:,:4] - np.tile(region[:2], 2)
        keep = np.all((rois[:,:4] >= 0) & 
                      (rois[:,:4] < np.tile(self.output_size, 2)), axis=1)
        rois = rois[keep,:]

        return {'image': image, 'ROI': rois}
#_______________________________________________________________________________

# ------------------------------------------------------------------------------
#   Randomized Image Horizontal Flip
# ------------------------------------------------------------------------------
class RandomHorizontalFlip(object):
    """Flip input image horizontally at random with a probability of 0.5"""

    # --------------------------------------------------------------------------
    def __call__(self, inputs):
        """Random image flip along Y axis"""
        image = inputs['image']
        rois  = inputs['ROI']
        
        if np.random.random_sample() > 0.5:
            # Flip if the sample drawn is larger than 0.5 in a uniform 
            # distribution range of [0. 1.)
            image = np.fliplr(image).copy()
            
            # Convert ROI coordinates:
            rois[:, [1,3]] = image.shape[1]-1 - rois[:, [3,1]]

        return {'image': image, 'ROI': rois}
#_______________________________________________________________________________

# ------------------------------------------------------------------------------
#   Randomized Image Vertical Flip
# ------------------------------------------------------------------------------
class RandomVerticalFlip(object):
    """Flip input image vertically at random with a probability of 0.5"""

    # --------------------------------------------------------------------------
    def __call__(self, inputs):
        """Random image flip along X axis"""
        image = inputs['image']
        rois  = inputs['ROI']
        
        if np.random.random_sample() > 0.5:
            # Flip if the sample drawn is larger than 0.5 in a uniform 
            # distribution range of [0. 1.)
            image = np.flipud(image).copy()
            
            # Convert ROI coordinates:
            rois[:, [0,2]] = image.shape[0]-1 - rois[:, [2,0]]

        return {'image': image, 'ROI': rois}
#_______________________________________________________________________________

# ------------------------------------------------------------------------------
#   Randomized Image Horizontal Flip
# ------------------------------------------------------------------------------
class JitterColor(object):
    """Wrapper around ColorJitter to handle target transformation"""

    # --------------------------------------------------------------------------
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.tform = ColorJitter(brightness=brightness,
                                 contrast=contrast,
                                 saturation=saturation,
                                 hue=hue)
    
    def __call__(self, inputs):
        """Calls colorjitter transform, np-op on targets"""
        image = inputs['image']
        rois  = inputs['ROI']
        image = self.tform.__call__(Image.fromarray(image))
        image = np.array(image)
        
        return {'image': image, 'ROI': rois}
#_______________________________________________________________________________

# ------------------------------------------------------------------------------
#   Converts sample image and bounding boxes to tensors.
# ------------------------------------------------------------------------------
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    # --------------------------------------------------------------------------
    def __init__(self, device='cpu'):
        self.device = device

    # --------------------------------------------------------------------------
    def __call__(self, inputs):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image  = inputs['image']
        image  = torch.from_numpy(image).to(self.device)
        image  = image.permute((2, 0, 1))
        rois   = inputs['ROI']
        rois   = torch.from_numpy(rois).to(self.device)
        
        return {'image': image, 'ROI': rois}
#_______________________________________________________________________________

# ------------------------------------------------------------------------------
#   Normalize image to unit normal intensities
# ------------------------------------------------------------------------------
class NormalizeIntensity(object):
    """Convert ndarrays in sample to Tensors."""
    
    # --------------------------------------------------------------------------
    def __init__(self, mean, std, device='cpu'):
        self.device = device
        self.Mean   = mean
        self.Std    = std

    # --------------------------------------------------------------------------
    def __call__(self, inputs):
        # Normalize to zero-mean normal intensities:
        image = inputs['image'].float()
        rois  = inputs['ROI']
        for n in range(image.size(0)):
            image[n,:,:] = (image[n,:,:] - self.Mean) / self.Std
        
        return {'image': image, 'ROI': rois}
#_______________________________________________________________________________
