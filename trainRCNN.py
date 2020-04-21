
from network.frcnn import FRCNN
import torch

image  = torch.zeros(5,3,640,640).float()
bbox   = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]])
bbox   = bbox.expand(5, -1, 4).contiguous()
labels = [0, 1]

fasterRCNNLayer = FRCNN(classes=labels)

# Forward pass:
prop, pred, clsprob, rpn_boxl, rpn_clsl, bboxl, clsl, labels = \
    fasterRCNNLayer(image, bbox)

print(prop.size())
print(clsprob.size())

"""
FRCNN - Train a Faster R-CNN model using the Open AI data challenge dataset.



import torch
from torch import nn
import torchvision
import numpy as np
import itertools
from torch import tensor
import bigtiff as btif

# ------------------------------------------------------------------------------
# Lets begin

bim = btif.Bigtiff('data/tumor_separate.tif')

# Bounding boxes are defined in the [y1, x1, y2, x2]
image = torch.zeros(1,3,800,800).float()
bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) 

# List VGG-16 layers
model  = torchvision.models.vgg16(pretrained=True)
layers = list(model.features)
layers = layers[0:len(layers)-1] # Removing the last max pooling layer.

# Conver the layers to a sequential module:
VGG16FeatureExtractor = nn.Sequential(*layers)
featureMap = VGG16FeatureExtractor(image)

# ------------------------------------------------------------------------------
# All images are down-sampled by 16 when passed through the network, so while
# creating bounding boxes for region proposals on the feature map, the box
# should be creating on the input image coordinates at the centre locations of 
# 16-by-16 image pixels which map to each pixel in the feature map.

# Create the anchors for each feature map location:
sampleRate = 16
_locations = np.arange(sampleRate/2, image.size()[3], sampleRate).astype(int)
centres    = np.array(np.meshgrid(_locations, _locations)).T.reshape(-1,2)

# Create anchors for each centre:
anchorScales   = [8, 16, 32]
anchorRatios   = [0.5, 1, 2]
anchorDims = [[sampleRate * _s * np.sqrt(_r), sampleRate * _s * np.sqrt(1./_r)]\
              for _s in anchorScales for _r in anchorRatios]

anchors = np.empty((0,4), dtype=np.float32)
for _dim in anchorDims:
    anchors = np.append(anchors, 
                        np.append(centres - np.multiply(0.5, _dim),
                                  centres + np.multiply(0.5, _dim), 
                                  axis=1),
                        axis=0)

# For training, Faster RCNN only takes into account the set of valid anchors
# that lie completely within the image, therefore, bound these anchors to 
# get the valid subset.
validAnchorInd = np.where((anchors[:,0] >= 0) & 
                          (anchors[:,1] >= 0) &
                          (anchors[:,2] <= image.size()[2]) &
                          (anchors[:,3] <= image.size()[3])
                         )[0]
validAnchors = anchors[validAnchorInd]

# ------------------------------------------------------------------------------
def IoU(anchors, bbox):
    # Function to calculate the Intersection Over Union of a set of anchors
    # against the set of ground truth bounding boxes specified in the inputs.
    
    # Initialize IoU matrix. The (i,j)_th element of this matrix corresponds 
    # to the IoU of the i_th anchor with the j_th ground truth bounding box.
    _iou = np.zeros((np.size(anchors, axis=0), np.size(bbox, axis=0)), 
                    dtype=np.float32)
    
    for _aid, _anchor in enumerate(anchors):
        # Find union coordinates and compute area:
        _union = np.array([np.minimum(bbox[:,0], _anchor[0]),
                           np.minimum(bbox[:,1], _anchor[1]),
                           np.maximum(bbox[:,2], _anchor[2]),
                           np.maximum(bbox[:,3], _anchor[3])]).transpose()
        _unionArea = np.multiply(_union[:,2] - _union[:,0],
                                 _union[:,3] - _union[:,1])
        
        # Find intersection coordinates and compute area:
        _intersection = np.array([
                           np.maximum(bbox[:,0], _anchor[0]),
                           np.maximum(bbox[:,1], _anchor[1]),
                           np.minimum(bbox[:,2], _anchor[2]),
                           np.minimum(bbox[:,3], _anchor[3])]).transpose()
        _interArea = np.multiply(_intersection[:,2] - _intersection[:,0],
                                 _intersection[:,3] - _intersection[:,1])
        
        # NOTE: Not all bounding boxes intersect the anchor, so we trim the ones
        #       that don't to have 0 intersection area.
        _interArea = np.clip(_interArea, 0, None)
        
        # Add computed values to the IoU matrix:
        _iou[_aid, :] = np.divide(_interArea, _unionArea)
    
    # Done, return the IoU matrix.
    return _iou

# ------------------------------------------------------------------------------
def classifyAnchors(iou):
    # Assigns "has-object" positive and "doesn't have object" negative labels
    # to each anchor based on computed IoU matrix:
    
    # Define thresholds on IoU for assigning positive and negative labels.
    _posthr = 0.7
    _negthr = 0.3
    
    # Initialize the label matrix with zeros (neutral label). Anchors labeled
    # 0 at the end of labeling are ignored and not considered for training.
    labels = np.zeros((np.size(iou, axis=0),), dtype=np.float32)
    
    _argmax_bbox = np.argmax(iou, axis=0)
    _max_iou     = np.amax(iou, axis=1)
    
    # CASE (a): The anchors with the highest intersection-over-union (IoU)
    # overlap with a ground-truth bounding box are positive.
    labels[_argmax_bbox] = 1
    
    # CASE (b): The anchors with the IoU overlap higher than the positive
    # threshold 0.7 with any ground-truth bounding box are assigned positively.
    labels[_max_iou >= _posthr] = 1
    
    # CASE (b): The anchors whose maximum IoU overlap remains lower than the
    # negative class threshold of 0.3 are assigned a negative class label.
    labels[_max_iou < _negthr] = -1
    
    return labels
    
    
# Classify valid labels based on their overlap with ground-truth bounding boxes.
iou          = IoU(validAnchors, bbox)
_labels      = classifyAnchors(iou)
anchors_pos  = np.where(_labels ==  1)[0]
anchors_neg  = np.where(_labels == -1)[0]

# Sample positive and negative anchors randomly upto a ratio of 1:1. If
# positive labels are less than sampled negative ones, pad with negative labels
# till a batch size of 256 is achieved.
batchSize   = 256
numPositive = np.clip(0.5*batchSize, None, len(anchors_pos))
_positive   = np.random.choice(anchors_pos, numPositive, replace=False)
_negative   = np.random.choice(anchors_neg, batchSize-numPositive,replace=False)

# NOTE: Convert anchor boxes and ground truth bounding boxes to 
# [Centre-y, Centre-x, Height, Width] format. This will help compute the
# bounding box regressions.
eps = np.finfo(validAnchors.dtype).eps # Bound h & w to avoid zero divisions.
centreAnchor = np.column_stack((
                    np.mean(validAnchors[:,[0,2]], axis=1),
                    np.mean(validAnchors[:,[1,3]], axis=1),
                    np.maximum(validAnchors[:,2] - validAnchors[:,0], eps),
                    np.maximum(validAnchors[:,3] - validAnchors[:,1], eps)
                ))

centreBbox   = np.column_stack((
                    np.mean(bbox[:,[0,2]], axis=1),
                    np.mean(bbox[:,[1,3]], axis=1),
                    np.maximum(bbox[:,2] - bbox[:,0], eps),
                    np.maximum(bbox[:,3] - bbox[:,1], eps)
                ))

# Calculate location regressions:
_dy = (centreBbox[:,0] - centreAnchor[:,0]) / centreAnchor[:,2]
_dx = (centreBbox[:,1] - centreAnchor[:,1]) / centreAnchor[:,3]
_dh = np.log(centreBbox[:,2] / centreAnchor[:,2])
_dw = np.log(centreBbox[:,3] / centreAnchor[:,3])

# Get final anchor locations and corresponding labels:
anchorLocs   = np.zeros((len(anchors,)) + anchors.shape[1:], dtype=np.float32)
anchorLabels = np.zeros((len(anchors,)), dtype=np.float32)
anchorLocs[validAnchorInd, :] = np.vstack((_dy, _dx, _dh, _dw)).transpose()
anchorLabels[validAnchorInd]  = _labels


# ------------------------------------------------------------------------------
#   Region Proposals Network
# ------------------------------------------------------------------------------

_rpn_depth     = 512
_feature_depth = featureMap.size()[1]
_num_anchor    = 9

# Create convolutional layer:
rpn_conv  = nn.Conv2d(_feature_depth, _rpn_depth, 3, stride=1, padding=1)
rpn_reg   = nn.Conv2d(_feature_depth, _num_anchor * 4, 1, stride=1, padding=0)
rpn_class = nn.Conv2d(_feature_depth, _num_anchor * 2, 1, stride=1, padding=0)

# Initialize all layers with weights drawn from a zero-mean Gaussian
# Distribution with a standard deviation of 0.01
rpn_conv.weight.data.normal_(0,0.01)
rpn_conv.bias.data.zero_()

rpn_reg.weight.data.normal_(0,0.01)
rpn_reg.bias.data.zero_()

rpn_class.weight.data.normal_(0,0.01)
rpn_class.bias.data.zero_()
"""