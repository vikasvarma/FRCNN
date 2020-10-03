"""
    Utility Tools for Faster R-CNN Object Detection Network
"""

import numpy as np
import torch
import torchvision
from config import Config

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
def box2targets(source, dest):
    """
        Computes parameterized coefficients between bounding boxes and targets. Assumes that bbox1 & bbox2 (target) are in [y1, x1, y2, x2] format.
    """
    
    ctrhwsrc = centrehw(source)
    ctrhwdst = centrehw(dest)
    
    # Calculate target coefficients:
    _dy = (ctrhwdst[:,:,0] - ctrhwsrc[:,:,0]) / ctrhwsrc[:,:,2]
    _dx = (ctrhwdst[:,:,1] - ctrhwsrc[:,:,1]) / ctrhwsrc[:,:,3]
    _dh = torch.log(ctrhwdst[:,:,2] / ctrhwsrc[:,:,2])
    _dw = torch.log(ctrhwdst[:,:,3] / ctrhwsrc[:,:,3])
        
    _coeff = torch.stack((_dy, _dx, _dh, _dw), dim=2)
        
    return _coeff

#-------------------------------------------------------------------------------
def targets2box(target, anchor):
    """
        Transform target coefficients computed wrt anchors to bounding boxes. Assumes that the anchors supplied are in [ctry, ctrx, h, w] format.
    """
    
    assert(target.size() == anchor.size())
    
    ctrhwAnchor = centrehw(anchor)
    
    # Calculate bounding boxes from targets:
    _ctry = (target[:,:,0] * ctrhwAnchor[:,:,2]) + ctrhwAnchor[:,:,0]
    _ctrx = (target[:,:,1] * ctrhwAnchor[:,:,3]) + ctrhwAnchor[:,:,1]
    _h    = ctrhwAnchor[:,:,2] * torch.exp(target[:,:,2])
    _w    = ctrhwAnchor[:,:,3] * torch.exp(target[:,:,3])
    
    ctrhw_bbox = torch.stack((_ctry, _ctrx, _h, _w), dim=2)
    _bbox      = yxcord(ctrhw_bbox)
    
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
        anchors = anchors.expand(bboxes.size(0),anchors.size(0),4).contiguous()
        
    assert(anchors.dim() == bboxes.dim())
    
    # Get batch sizes, number of bounding boxes and number of anchors
    _b, _N, _ = bboxes.size()
    _K = anchors.size(1)
    
    # Initialize IoU matrix:
    _iou = torch.zeros((_b, _K, _N), dtype=bboxes.dtype)
    
    # Compute area of anchors and bboxes:
    _anchorW = (anchors[:,:,2] - anchors[:,:,0] + 1)
    _anchorH = (anchors[:,:,3] - anchors[:,:,1] + 1)
    _anchorA = (_anchorW * _anchorH).view(_b, _K, 1)
    _anchor0 = ((_anchorW == 1) * (_anchorH == 1)).view(_b, _K, 1)
    
    _bboxesW = (bboxes[:,:,2] - bboxes[:,:,0] + 1)
    _bboxesH = (bboxes[:,:,3] - bboxes[:,:,1] + 1)
    _bboxesA = (_bboxesW * _bboxesH).view(_b, 1, _N)
    _bboxes0 = ((_bboxesW == 1) * (_bboxesH == 1)).view(_b, 1, _N)
    
    # Expand anchors and bounding boes to match sizes:
    # NOTE: Use expand instead of repeat, expanding singleton dimensions
    #       conserves memory in pytorch, and is faster.
    anchors   = anchors.view(_b,_K,1,4).expand(_b,_K,_N,4).contiguous()
    bboxes    = bboxes.view(_b,1,_N,4).expand(_b,_K,_N,4).contiguous()
        
    # Find intersection coordinates and compute area:
    # Size of intersection area: [b K N]
    _interW   = torch.min(bboxes[:,:,:,2], anchors[:,:,:,2]) - \
                torch.max(bboxes[:,:,:,0], anchors[:,:,:,0]) + 1
    _interH   = torch.min(bboxes[:,:,:,3], anchors[:,:,:,3]) - \
                torch.max(bboxes[:,:,:,1], anchors[:,:,:,1]) + 1
        
    # NOTE: Not all bounding boxes intersect the anchor, so we trim the ones
    #       that don't to have 0 intersection area.
    _interA = torch.clamp(_interW, min=0) * torch.clamp(_interH, min=0)
    
    # Compute Intersection-Over-Union and return.
    _iou    = _interA / (_anchorA + _bboxesA - _interA)
    
    # Clamp the overlap area for single pixel ROIs:
    _iou.masked_fill_(_bboxes0.expand(_b, _K, _N),  0)
    _iou.masked_fill_(_anchor0.expand(_b, _K, _N), -1)
    
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
    pos_ind = torch.nonzero(_max_iou >= posthr, as_tuple=True)
    labels[pos_ind[0], pos_ind[1]] = 1
        
    # CASE (c): The anchors whose maximum IoU overlap remains lower than the
    #           negative class threshold (default = 0.3) are assigned a 
    #           negative class label.
    neg_ind = torch.nonzero(_max_iou <  negthr, as_tuple=True)
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
            
        # If enough samples are not available, sample from the ignored labels:
        total = torch.sum(labels[_batch].ne(-1))
        if total < numSamples:
            ignored = torch.nonzero(labels[_batch] == -1)
            labels[_batch, ignored[range(numSamples - total)]] = 0
            
    # Done, excess samples are assigned the ignore label.
    return labels


#-------------------------------------------------------------------------------
def nms(bboxes, scores, n_pre, n_post):
    """
    Non-Maximum Supression (NMS): Supresses capturing ROIs which point to the same object. This is identified through the IoU between predicted targets.
        
    Inputs  ____________________________________________________________________
            bboxes - [b N 4] Tensor
                     [y1 x1 y2 x2] bounding box coordinates
            scores - [b N] Tensor
                     Objectness scores of each bounding box
    """
    assert((bboxes.size(0) == scores.size(0)) & 
           (bboxes.size(1) == scores.size(1)))
    
    # NMS Parameters:
    nms_thr = Config.NMS_THR
    B = bboxes.size(0)
    
    # Initialize the output bounding box set:
    boxset = bboxes.new(torch.Size([B, n_post, 4])).zero_()
    
    # For each batch, apply the NMS threshold to get top N predictions:
    for _b in range(B):
        # Filter out proposals with either width or height less than minimum 
        # size threshold.
        boxw = bboxes[_b,:,2] - bboxes[_b,:,0] + 1
        boxh = bboxes[_b,:,3] - bboxes[_b,:,1] + 1
        min_size   = Config.MIN_BOX_SIZE
        valid_size = torch.nonzero((boxw >= min_size) & (boxh >= min_size))
        _boxes     = bboxes[_b, valid_size, :].squeeze_()
        _score     = scores[_b, valid_size].squeeze_()
        
        # Sort the scores and retain ones that are have the top N scores:
        _score, _order = torch.sort(_score, descending=True)
        
        # Sort batch boxes in descending order:
        _boxes = _boxes[_order, :]
        
        # Select the top performing N_PRE boxes:
        _boxes = _boxes[:min(_boxes.size(0), n_pre), :]
        _score = _score[:min(_score.size(0), n_pre)]
        
        # NMS and select top N candidates:
        _boxes = _boxes[:, [1,0,3,2]]
        _score = _score.type(_boxes.dtype)
        _keep  = torchvision.ops.nms(_boxes, _score, nms_thr)
        _keep  = _keep[:min(_keep.size(0), n_post)]
        
        # Flip boxes from XY -> YX coordinates and store in the boxset:
        keep_boxes = _boxes[_keep]
        boxset[_b, :_keep.numel(), :] = keep_boxes[:, [1,0,3,2]]
    
    return boxset

#-------------------------------------------------------------------------------
def smoothL1Loss(predictions, groundTruth, weights, sigma):
    """
    Smooth L1 Loss Function: Used to compute the regression loss in  predicted and ground truth targets.
        
    Inputs  ____________________________________________________________________
            predictions, groundTruth - [b N 4] Pytorch Variable corresponding
                                       to [ty tx th tw] coefficients
    """
    
    assert(predictions.size() == groundTruth.size())
    assert(weights.size()     == groundTruth.size())   
    
    sigma = sigma ** 2
    diff  = weights * (predictions - groundTruth)
    flag  = (diff.abs().data < (1. / sigma)).float()
    loss  = (flag * (sigma / 2.) * (diff ** 2) +
            (1 - flag) * (diff.abs() - 0.5 / sigma))
    
    return loss.sum()