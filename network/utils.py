"""
    Utility Tools for Faster R-CNN Object Detection Network
"""

import numpy as np
import torch
import torchvision
from .config import Config

#-------------------------------------------------------------------------------
def centrehw(roi):
    """
        Utility function to convert a list of ROI boxes specified in 
        [y1 x1 y2 x2] format to [ctry ctrx h w] tensors.
    """
    
    if roi.dim() == 2:
        roi = roi.expand(1, roi.size(0), roi.size(1)).contiguous()
    
    # Clip the minimum height and width to eps to ensure that zero divisions
    # are avoided.
    _eps = np.finfo(np.float).eps
    
    _ctrhw_roi = torch.stack((
                    torch.mean(roi[:,:,[0,2]], dim=2),
                    torch.mean(roi[:,:,[1,3]], dim=2),
                    torch.clamp(roi[:,:,2] - roi[:,:,0] + 1, min=_eps),
                    torch.clamp(roi[:,:,3] - roi[:,:,1] + 1, min=_eps)
                ), dim=2)
    
    return _ctrhw_roi

#-------------------------------------------------------------------------------
def yxcord(roi):
    """
        Utility function to convert a list of ROI boxes specified in 
        [ctry ctrx h w] format to [y1 x1 y2 x2] tensors.
    """
        
    _yx_roi = torch.stack((
                        roi[:,:,0] - 0.5 * roi[:,:,2],
                        roi[:,:,1] - 0.5 * roi[:,:,3],
                        roi[:,:,0] + 0.5 * roi[:,:,2],
                        roi[:,:,1] + 0.5 * roi[:,:,3]
                    ), dim=2)
    return _yx_roi

#-------------------------------------------------------------------------------
def targets(bbox, anchor):
    """
        Computes parameterized target coefficients wrt anchors. Assumes that the anchors supplied are in [ctry, ctrx, h, w] format.
    """
    
    # Calculate target coefficients:
    _dy = (bbox[:,:,0] - anchor[:,:,0]) / anchor[:,:,2]
    _dx = (bbox[:,:,1] - anchor[:,:,1]) / anchor[:,:,3]
    _dh = torch.log(bbox[:,:,2] / anchor[:,:,2])
    _dw = torch.log(bbox[:,:,3] / anchor[:,:,3])
        
    _coeff = torch.stack((_dy, _dx, _dh, _dw), dim=2)
        
    return _coeff

#-------------------------------------------------------------------------------
def invtargets(target, anchor):
    """
        Transform target coefficients computed wrt anchors to bounding boxes. Assumes that the anchors supplied are in [ctry, ctrx, h, w] format.
    """
    
    assert(target.size() == anchor.size())
    
    # Calculate bounding boxes from targets:
    _ctry = (target[:,:,0] * anchor[:,:,2]) + anchor[:,:,0]
    _ctrx = (target[:,:,1] * anchor[:,:,3]) + anchor[:,:,1]
    _h    = anchor[:,:,2] * torch.exp(target[:,:,2])
    _w    = anchor[:,:,3] * torch.exp(target[:,:,3])
    
    _bbox = torch.stack((_ctry, _ctrx, _h, _w), dim=2)
    
    return _bbox
    
#-------------------------------------------------------------------------------
def IoU(bboxes, anchors):
    """
    Function to calculate the Intersection Over Union of a set of anchors against the set of ground truth bounding boxes specified in the inputs.
    
    Inputs  ____________________________________________________________________
            BBOXES  - [b N 4] Tensor
            ANCHORS - [b K 4] Tensor
        
    Outputs ____________________________________________________________________
            IoU     - [b K N] Tensor
                      The (b,i,j)_th element of this matrix corresponds to the IoU of the i_th anchor with the j_th ground truth bounding box corresponding to the bth batch.
    """
    
    if anchors.dim() == 2:
        anchors = anchors.expand(_b, anchors.size(0), 4).contiguous()
        
    assert(anchors.dim() == bboxes.dim())
    
    # Get batch sizes, number of bounding boxes and number of anchors
    _b, _N, _ = bboxes.size()
    _K = anchors.size(1)
    
    # Initialize IoU matrix:
    _iou = torch.zeros((_b, _K, _N), dtype=bboxes.dtype)
    
    # Compute area of anchors and bboxes:
    _anchorArea = ((anchors[:,:,2] - anchors[:,:,0]) * \
                   (anchors[:,:,3] - anchors[:,:,1])).view(_b, _K, 1)
    _bboxesArea = ((bboxes[:,:,2] - bboxes[:,:,0]) * \
                   (bboxes[:,:,3] - bboxes[:,:,1])).view(_b, 1, _N)
    
    # Expand anchors and bounding boes to match sizes:
    # NOTE: Use expand instead of repeat, expanding singleton dimensions
    #       conserves memory in pytorch, and is faster.
    anchors   = anchors.view(_b,_K,1,4).expand(_b,_K,_N,4).contiguous()
    bboxes    = bboxes.view(_b,1,_N,4).expand(_b,_K,_N,4).contiguous()
        
    # Find intersection coordinates and compute area:
    # Size of intersection area: [b K N]
    _iwidth   = torch.min(bboxes[:,:,:,2], anchors[:,:,:,2]) - \
                torch.max(bboxes[:,:,:,0], anchors[:,:,:,0])
    _iheight  = torch.min(bboxes[:,:,:,3], anchors[:,:,:,3]) - \
                torch.max(bboxes[:,:,:,1], anchors[:,:,:,1])
        
    # NOTE: Not all bounding boxes intersect the anchor, so we trim the ones
    #       that don't to have 0 intersection area.
    _iarea    = torch.clamp(_iwidth * _iheight, min=0)
        
    # Compute Intersection-Over-Union and return.
    _iou      = _iarea / (_anchorArea + _bboxesArea - _iarea)
    return _iou

#-------------------------------------------------------------------------------
def classify(iou, posthr, negthr):
    """
    Assigns "has-object" positive and "doesn't have object" negative labels to each anchor based on computed IoU matrix:
        
        IOU - [b K N] Tensor       
            * b - BatchSize
            * K - Number of anchors
            * N - Number of bounding boxes.
    """
        
    _b, _K, _N           = iou.size()
    _max_iou, argmax_gt  = torch.max(iou, dim=2)    # Which GT has max overlap
    argmax_bbox          = torch.argmax(iou, dim=1) # Which bbox has max overlap
       
    # Initialize the label matrix with zeros (neutral label). Anchors
    # labeled -1 at the end of labeling are ignored and not considered for
    # training.
    labels = iou.new_full((_b, _K), fill_value=-1).int()
        
    # CASE (a): The anchors with the highest intersection-over-union (IoU)
    #           overlap with a ground-truth bounding box are positive.
    for _batch in range(_b):
        labels[_batch, argmax_bbox[_batch]] = 1
    
    # CASE (b): The anchors with the IoU overlap higher than the positive
    #           threshold (default = 0.7) with any ground-truth bounding
    #           box are assigned positively.
    pos_ind = torch.nonzero((_max_iou <= posthr[1]) &
                            (_max_iou >= posthr[0]), as_tuple=True)
    labels[pos_ind[0], pos_ind[1]] = 1
        
    # CASE (c): The anchors whose maximum IoU overlap remains lower than the
    #           negative class threshold (default = 0.3) are assigned a 
    #           negative class label.
    neg_ind = torch.nonzero((_max_iou <  negthr[1]) &
                            (_max_iou >= negthr[0]), as_tuple=True)
    labels[neg_ind[0], neg_ind[1]] = 0
    
    return labels, argmax_gt

#-------------------------------------------------------------------------------
def randsample(labels, numSamples, sampleRatio):
    """
        TODO
    """
        
    # Calculate number of positive and negative labels for each batch:
    _req_pos     = int(numSamples * sampleRatio)
    _numPositive = torch.sum((labels == 1).int(), 1)
    _numNegative = torch.sum((labels == 0).int(), 1)
            
    for _batch in range(labels.size(0)):
        # For each batch, randomly disable samples if their total exceeds whats
        # required to construct the sample.
        if _numPositive[_batch] > _req_pos:
            _posind = torch.nonzero(labels[_batch] == 1).view(-1)
            _disind = _posind[torch.randperm(_posind.size(0))
                                    [:_posind.size(0) - _req_pos]]
            labels[_batch, _disind] = -1
            
        # Handle negative samples based on the number of positive labels:
        _req_neg = numSamples - torch.sum(labels[_batch] == 1)
        
        if _numNegative[_batch] > _req_neg:
            _negind = torch.nonzero(labels[_batch] == 0).view(-1)
            _disind = _negind[torch.randperm(_negind.size(0))
                                    [:_negind.size(0) - _req_neg]]
            labels[_batch, _disind] = -1
            
    # Done, excess samples are assigned the ignore label.
    return labels


#-------------------------------------------------------------------------------
def nms(bboxes, scores):
    """
    Non-Maximum Supression (NMS): Supresses capturing ROIs which point to the same object. This is identified through the IoU between predicted targets.
        
    Inputs  ____________________________________________________________________
            bboxes - [b N 4] Tensor
                     [y1 x1 y2 x2] bounding box coordinates
            scores - [b N] Tensor
                     Objectness scores of each bounding box
    """
    assert(bboxes.size(1) == scores.size(1))
    
    # NMS Parameters:
    if Config.TRAIN.TRAINING:
        pre_nms_N  = Config.RPN.PRE_TRAIN_NMS_N
        post_nms_N = Config.RPN.POST_TRAIN_NMS_N
    else:
        pre_nms_N  = Config.RPN.PRE_TEST_NMS_N
        post_nms_N = Config.RPN.POST_TEST_NMS_N
            
    nms_thr    = Config.RPN.NMS_THR
    _batchSize = bboxes.size(0)
       
    # Sort the scores and retain ones that are have the top N scores:
    scores, _order = torch.sort(scores, dim=1, descending=True)
    bboxes = bboxes[:, _order.view(-1), :]
    boxset = bboxes.new(_batchSize, post_nms_N, 4).zero_()
        
    for _batch in range(_batchSize):
        # For each batch, apply the NMS threshold on top N predictions:
        _boxes = bboxes[_batch, :min(bboxes.size(1), pre_nms_N), :]
        _score = scores[_batch, :min(scores.numel(), pre_nms_N)]
        
        # NMS and select top N candidates:
        _boxes = _boxes.view(-1, 4)[:, [1,0,3,2]]
        _score = _score.view(-1).type(_boxes.dtype)
        _keep  = torchvision.ops.nms(_boxes, _score, nms_thr)
        _keep  = _keep[:min(_keep.size(0), post_nms_N)]
            
        boxset[_batch, :_keep.numel(), :] = _boxes[_keep, :] 
    
    return boxset

#-------------------------------------------------------------------------------
def smoothL1Loss(predictions, groundTruth, inweights, outweights):
    """
    Smooth L1 Loss Function: Used to compute the regression loss in  predicted and ground truth targets.
        
    Inputs  ____________________________________________________________________
            predictions, groundTruth - [b N 4] Pytorch Variable corresponding
                                       to [ty tx th tw] coefficients
    """
    
    assert(predictions.size() == groundTruth.size())
    assert(inweights.size()   == groundTruth.size())
    assert(outweights.size()  == groundTruth.size())   
     
    # L1 Loss Paramters:
    sigma = 6
        
    difference = predictions - groundTruth
    in_diff    = torch.abs(inweights * difference)
    smoothL1   = (in_diff < 1. / sigma).detach().float()
    in_loss    = torch.pow(in_diff,2) * (sigma / 2.0) * smoothL1 + \
                 (in_diff - (0.5 / sigma)) * (1. - smoothL1)
    loss       = outweights * in_loss
        
    # Find the total loss as an average across all dimensions:
    for _d in sorted(range(loss.dim()), reverse=True):
        loss = loss.sum(dim=_d)
    
    loss = loss.mean()
            
    return loss