"""
Module Information
    RPN - Region Proposals Network
"""

# All import statements go here:
import torch
from torch import nn
from torch.nn import functional as fcn
import numpy as np
from .config import Config
from .utils import *

#-------------------------------------------------------------------------------
#  Region Proposal Layer
#-------------------------------------------------------------------------------
class ProposalLayer(nn.Module):
    """
        Assigns anchors to ground truth bounding boxes, classifies anchor objectness and produces bounding-box regression targets.
    """
    
    #---------------------------------------------------------------------------
    def __init__(self, input_size):
        super(ProposalLayer, self).__init__()
        
        self._stride   = Config.RPN.SPATIAL_SCALE
        self._scales   = Config.RPN.ANCHOR_SCALES
        self._ratios   = Config.RPN.ANCHOR_RATIOS
        self.InputSize = input_size
        self.NumAnchor = len(self._scales) * len(self._ratios)
        self._generate_anchors()

    #---------------------------------------------------------------------------
    def _generate_anchors(self):
        """
            All images are down-sampled by FeatureStride (s) when passed through VGG network, so while creating bounding boxes for region proposals on the feature map, the box should be creating on the input image coordinates at the centre locations of s-by-s image pixels which map to each pixel in the feature map.
        """

        # Create the anchors for each feature map location:
        _xypoints  = np.arange(self._stride/2, self.InputSize[3] * self._stride,
                               self._stride).astype(int)
        centres    = np.array(np.meshgrid(_xypoints, _xypoints)).T.reshape(-1,2)

        # Create anchors for each centre:
        _anchorDims = [[self._stride * _s * np.sqrt(_r), 
                        self._stride * _s * np.sqrt(1./_r)]\
                        for _s in self._scales for _r in self._ratios]

        _anchors = np.empty((0,4), dtype=np.float)
        for _dim in _anchorDims:
            _anchors = np.append(_anchors, 
                                np.append(centres - np.multiply(0.5, _dim),
                                          centres + np.multiply(0.5, _dim), 
                                          axis=1), axis=0)

        # Convert anchors to torch tensors:
        self.Anchors      = torch.from_numpy(_anchors)
        self.CtrHWAnchors = centrehw(self.Anchors)
        
        # For training, Faster RCNN only takes into account the set of valid
        # anchors that lie completely within the image, therefore, bound these
        # anchors to get the valid subset.
        self._keepAnchors = (
                    (self.Anchors[:,0] >= 0) &
                    (self.Anchors[:,1] >= 0) &
                    (self.Anchors[:,2] <= self.InputSize[2] * self._stride) &
                    (self.Anchors[:,3] <= self.InputSize[3] * self._stride))
        self._insideInd   = torch.nonzero(self._keepAnchors).view(-1)
    
    #---------------------------------------------------------------------------
    def forward(self, pred_boxes, cls_scores, gt_boxes):
        """
        """
        _batchSize, _, H, W = pred_boxes.size()
        _A = self.NumAnchor
        
        # NOTE: We only care about foreground class probabilities, so trim the
        #       cls_scores to contain only foreground entries. This is the
        #       later half of the stack along dim=1.
        pos_scores = cls_scores[:, int(cls_scores.size(1)/2):, :, :]
        
        # Rehape predicted bounding-boxes and class scores to [b K*N _] blobs:
        # NOTE: Before reshaping, its important to permute the dimensions of 
        #       these tensors in order to make the third dimension store the 
        #       predictions for each pixel in the feature map.
        pred_boxes = pred_boxes.permute(0,2,3,1).contiguous()
        pred_boxes = pred_boxes.view(_batchSize, -1, 4)
        
        cls_scores = cls_scores.permute(0,2,3,1).contiguous()
        cls_scores = cls_scores.view(_batchSize, -1, 2)
        
        pos_scores = pos_scores.permute(0,2,3,1).contiguous()
        pos_scores = pos_scores.view(_batchSize, -1)
        
        # Initialize RPN losses:
        self.rpn_bbox_loss = 0
        self.rpn_cls_loss  = 0
        
        # Get anchors and expand to scale to batch size:
        _batchSize   = pred_boxes.size(0)
        _anchorSize  = self.Anchors.size(0)
        
        anchors      = self.Anchors.expand(_batchSize, _anchorSize, 4)
        anchors      = anchors.type(gt_boxes.dtype).to(gt_boxes.device)
        ctrhwAnchors = self.CtrHWAnchors.expand(_batchSize, _anchorSize, 4)
        ctrhwAnchors = ctrhwAnchors.type(gt_boxes.dtype).to(gt_boxes.device)
        
        # Predicted Top Bounding Box Proposals: ________________________________________________________________________
        # Convert proposals from target coefficients to bounding box
        # coordinates:
        _ctrhwBbox  = invtargets(pred_boxes, ctrhwAnchors)
        proposals   = yxcord(_ctrhwBbox)
        
        # Clamp the proposals to the image extents:
        proposals[:,:,0].clamp_(min=0, max=(self.InputSize[2] * self._stride)-1)
        proposals[:,:,1].clamp_(min=0, max=(self.InputSize[3] * self._stride)-1)
        proposals[:,:,2].clamp_(min=0, max=(self.InputSize[2] * self._stride)-1)
        proposals[:,:,3].clamp_(min=0, max=(self.InputSize[3] * self._stride)-1)
        
        # Apply non-maximal supression to obtain top scoring targets:
        top_proposals = nms(proposals, pos_scores, self.training)
        top_proposals = torch.floor(top_proposals)
        
        # TRAINING: Obtain Ground Truth Bounding Box Targets and use these
        #           to compute RPN bbox prediction and classification loss._______________________________________________________________________
        if self.training:
            # Compute ground truth boxes overlap with anchors and label the
            # anchors. Only consider anchors that are completely inside the 
            # image for training.
            _iou    = IoU(gt_boxes, anchors)
            _labels, _argmax_bbox = classify(_iou, 
                                             Config.RPN.IOU_POSITIVE_THR,
                                             Config.RPN.IOU_NEGATIVE_THR)
            
            # Ignore out of bound anchors:
            _labels[:, self._insideInd] = -1;
            
            # Randomly sample a fixed subset of the anchors:
            _labels = randsample(_labels, Config.RPN.ANCHOR_SAMPLES,
                                          Config.RPN.ANCHOR_SAMPLE_RATIO)
            
            # Calculate ground-truth targets:
            # NOTE: Select only the ground truth objects which have the maximum
            #       overlap with each anchor to compute the target coefficients.
            gt_bbox_targets = gt_boxes.new(anchors.size()).zero_()
            for _batch in range(_batchSize):
                gt_bbox_targets[_batch] = gt_boxes[_batch, _argmax_bbox[_batch]]
                
            gt_bbox_targets = centrehw(gt_bbox_targets)
            gt_bbox_targets = targets(gt_bbox_targets, ctrhwAnchors)
            gt_bbox_targets = gt_bbox_targets.requires_grad_()
            
            # Calculate RPN Losses: ____________________________________________________________________
            # 1. Classification Loss:
            #    Use the class scores input and the computed objectness
            #    classification labels to compute the cross-entropy loss
            #    between prediction and ground truth labels.
            
            # Locate the sampled labels from class scores:
            _keep      = _labels.view(-1).ne(-1).nonzero().view(-1)
            keep_score = torch.index_select(cls_scores.view(-1, 2), 0, _keep)
            rpn_labels = torch.index_select(_labels.view(-1), 0, _keep.data)
            rpn_labels = rpn_labels.long()
            
            self.rpn_cls_loss = fcn.cross_entropy(keep_score, rpn_labels)
            
            # 2. Regression Loss:
            #    Use the bounding box targets computed for ground truth boxes
            #    with largest overlap with the anchors in computing the
            #    regression loss against predicted targets using a smooth L1l
            #    oss function.
            
            # Create weights used in measuring L1 loss:
            # NOTE: Along the last dimension, the two weights correspond to 
            #       inside and outside bbox weights respectively.
            in_weight  = gt_boxes.new(_batchSize, anchors.size(1)).zero_()
            out_weight = gt_boxes.new(_batchSize, anchors.size(1)).zero_()
            
            in_weight[_labels == 1]   = Config.RPN.L1WEIGHT_INTERSECTION
            out_weight[_labels != -1] = (1.0/torch.sum(_labels[0] >= 0).float()).to(out_weight.dtype)
            
            # Reshape weights to match feature sizes:
            in_weight  = in_weight.view(_batchSize, anchors.size(1), 1)
            in_weight  = in_weight.expand(_batchSize, anchors.size(1), 4)
            in_weight  = in_weight.contiguous().view(_batchSize, H, W, 4*_A)\
                                    .permute(0,3,1,2).contiguous()
            
            out_weight = out_weight.view(_batchSize, anchors.size(1), 1)
            out_weight = out_weight.expand(_batchSize, anchors.size(1), 4)
            out_weight = out_weight.contiguous().view(_batchSize, H, W, 4*_A)\
                                    .permute(0,3,1,2).contiguous()
            
            # Reshape predictions, weights and targets to sizes equivalent to 
            # input sizes.
            pred_boxes      = pred_boxes.view(_batchSize,H,W,_A*4).contiguous()
            pred_boxes      = pred_boxes.permute(0,3,1,2).contiguous()
            gt_bbox_targets = gt_bbox_targets.view(_batchSize,H,W,_A*4)\
                                             .contiguous()
            gt_bbox_targets = gt_bbox_targets.permute(0,3,1,2).contiguous()
            
            self.rpn_bbox_loss = smoothL1Loss(pred_boxes,
                                              gt_bbox_targets, 
                                              in_weight,
                                              out_weight)

        # Return the top proposals and computed losses: ________________________________________________________________________
        return top_proposals, self.rpn_bbox_loss, self.rpn_cls_loss
#_______________________________________________________________________________

#-------------------------------------------------------------------------------
#  Region Proposal Network
#-------------------------------------------------------------------------------
class RPN(nn.Module):
    """
        Region Proposal Network: Class which contains utility methods to compute the region proposals of an image.
    """
    
    def __init__(self, feature_size):
        super(RPN, self).__init__()
        
        # Define network and anchor parameters:
        self._rpn_depth    = Config.RPN.FEATURE_DEPTH
        self.FeatureSize   = feature_size
        self._num_anchor   = len(Config.RPN.ANCHOR_SCALES) * \
                             len(Config.RPN.ANCHOR_RATIOS)
        
        # Initialize the network:
        self._initialize_network()
    
    def _initialize_network(self):
        # Create convolutional layers:
        self.RPNConvLayer           = nn.Conv2d(self.FeatureSize[1], 
                                                self._rpn_depth, 
                                                kernel_size=3, 
                                                stride=1, 
                                                padding=1)
        
        self.RPNRegressionLayer     = nn.Conv2d(self._rpn_depth, 
                                                self._num_anchor * 4, 
                                                kernel_size=1, 
                                                stride=1, 
                                                padding=0)
        
        self.RPNClassificationLayer = nn.Conv2d(self._rpn_depth, 
                                                self._num_anchor * 2, 
                                                kernel_size=1, 
                                                stride=1, 
                                                padding=0)

        self.RPNProposalLayer       = ProposalLayer(self.FeatureSize)

        # Initialize all layers with weights drawn from a zero-mean Gaussian
        # Distribution with a standard deviation of 0.01.
        self.RPNConvLayer.weight.data.normal_(0,0.01)
        self.RPNConvLayer.bias.data.zero_()
        self.RPNRegressionLayer.weight.data.normal_(0,0.01)
        self.RPNRegressionLayer.bias.data.zero_()
        self.RPNClassificationLayer.weight.data.normal_(0,0.01)
        self.RPNClassificationLayer.bias.data.zero_()
    
    @staticmethod
    def _reshape_(tensor, new_shape):
        _shape = tensor.size()
        tensor = tensor.view(_shape[0],
                             int(new_shape),
                             int(float(_shape[1]*_shape[2]) / float(new_shape)),
                             _shape[3])
        return tensor
        
    def forward(self, feature_map, gt_boxes):
        """
        TODO
        
        Forward propagation fed into by the VGG feature extractor.
        
        """

        # Get the size of the feature map
        _batchSize, _depth, _width, _height = feature_map.size()
        
        # Scan the feature map through the conv-relu layer:
        _rpn_conv_map = fcn.relu(self.RPNConvLayer(feature_map), inplace=True)
        
        # Using the convoluted features, derive classification and offset
        # regressions to bounding box predictions.
        _rpn_class_score = self.RPNClassificationLayer(_rpn_conv_map)
        _rpn_class_prob  = fcn.softmax(self._reshape_(_rpn_class_score,2), 1)
        _rpn_class_prob  = self._reshape_(_rpn_class_prob, self.RPNClassificationLayer.out_channels)
        
        # offsets to anchors:
        _rpn_bbox_pred   = self.RPNRegressionLayer(_rpn_conv_map)
        
        # Compute region proposals from the proposals layer:
        proposals, bbox_loss, cls_loss = self.RPNProposalLayer(
            _rpn_bbox_pred, _rpn_class_prob, gt_boxes)
        
        return proposals, bbox_loss, cls_loss
#_______________________________________________________________________________
    