"""
Module Information
    FRCNN - Faster R-CNN Network
"""

# NOTE: This implementation of the Faster R-CNN Object Detection network
#       utilizes the ROI layers module from the work done by jwyang in 
#       https://github.com/jwyang/faster-rcnn.pytorch.git.
#
# Add the path to ROI layers
# import sys
# sys.path.append("C:\Projects\faster-rcnn.pytorch\lib")

from   torch.autograd   import Variable
from   torch            import nn
from   model.roi_layers import ROIPool
from   .rpn             import RPN
from   .config          import Config
from   .utils           import *
from   torchvision.ops  import roi_pool
import torch
import torchvision
import torch.nn.functional as fcn

#-------------------------------------------------------------------------------
#  Proposal Target Layer
#-------------------------------------------------------------------------------
class ProposalTargetLayer(nn.Module):
    """
        Assigns ROI proposals to ground-truth boxes, samples a batch of ROI proposals, classifies them and computes ground-truth targets.
    """
    
    def __init__(self, classes):
        super(ProposalTargetLayer, self).__init__()
        self.Classes = classes
    
    def forward(self, proposals, gt_boxes):
        """
            TODO 
            
            Returns:
                TODO
        """
        
        _batchSize = proposals.size(0)
        
        # Compute the overlap between the proposals and ground-truth:
        _iou = IoU(gt_boxes, proposals)
        _labels, _argmax_gt = classify(_iou, Config.FRCNN.ROI_POSITIVE_THR,
                                             Config.FRCNN.ROI_NEGATIVE_THR)
        
        # Randomly sample a fixed subset of proposals:
        _sample_size = Config.FRCNN.ROI_SAMPLES
        batch_labels = randsample(_labels, _sample_size,
                                  Config.FRCNN.ROI_SAMPLE_RATIO)
        _keep        = torch.nonzero(batch_labels.ne(-1), as_tuple=True)
        
        # Extract ROI proposals batch and its corresponding objectness labels.
        roi_batch    = proposals[_keep[0], _keep[1], :]
        roi_batch    = roi_batch.view(_batchSize, _sample_size, 4).contiguous()
        batch_labels = batch_labels[batch_labels >= 0]
        batch_labels = batch_labels.view(_batchSize, _sample_size)
        
        # Create a batch set of ground truth objects which have the max overlap
        # wrt sampled batches.
        _argmax_gt   = _argmax_gt[_keep[0], _keep[1]]
        _argmax_gt   = _argmax_gt.view(_batchSize, _sample_size).contiguous()
        gt_batch     = proposals.new(roi_batch.size()).zero_()
        for _b in range(_batchSize):
            gt_batch[_b] = gt_boxes[_b, _argmax_gt[_b], :]
        
        # Compute regression targets for sample batch:
        bbox_targets = targets(centrehw(gt_batch), centrehw(roi_batch))
        
        # Compute weights used in estimating loss:
        in_weights  = gt_boxes.new(bbox_targets.size()).zero_()
        in_weights[batch_labels == 1] = Config.RPN.L1WEIGHT_INTERSECTION
        out_weights = (in_weights > 0).float()
        
        return roi_batch, batch_labels, bbox_targets, in_weights, out_weights
#_______________________________________________________________________________

#-------------------------------------------------------------------------------
#  Faster R-CNN Object Detection Network
#-------------------------------------------------------------------------------
class FRCNN(nn.Module):
    """
        Faster R-CNN Network
    """
    
    def __init__(self, classes):
        # Initialize the components of the Faster R-CNN network.
        super(FRCNN, self).__init__()
        
        self.Classes = classes
        
        # Extract the pretrained convolutional layers from a VGG-16 network
        # which will be used as the base feature extractor.
        vgg = torchvision.models.vgg16(pretrained=True)
        self.VGG16FeatureExtractor  = nn.Sequential(*list(vgg.features)[:-1])
        self.VGG16FeatureClassifier = nn.Sequential(*list(vgg.classifier)[:-1])
        
        # Initialize the RPN network layer which provides region proposals from
        # computed convolutional features
        rpn_feat_size = torch.Tensor(Config.FRCNN.FEATURE_SIZE).clone().int()
        rpn_feat_size[1]    = Config.RPN.FEATURE_DEPTH
        rpn_feat_size[2:4] /= Config.FRCNN.SPATIAL_SCALE
        self.RPNLayer       = RPN(rpn_feat_size)
        
        # Initialize the proposal target layer for training:
        self.PropTargetLayer = ProposalTargetLayer(self.Classes)
        
        # Initialize the fully-connected layers:
        self.FRCNNClassScore = nn.Linear(4096, len(self.Classes))
        self.FRCNNBBoxPred   = nn.Linear(4096, 4)
        
        # Initialize all FC layers with weights drawn from a zero-mean Gaussian
        # Distribution with a standard deviation of 0.01.
        self.FRCNNClassScore.weight.data.normal_(0,0.01)
        self.FRCNNClassScore.bias.data.zero_()
        self.FRCNNBBoxPred.weight.data.normal_(0,0.01)
        self.FRCNNBBoxPred.bias.data.zero_()
    
    
    def forward(self, batch, gt_boxes):
        """
            Feed forward the batch of images through the network to predict object bounding boxes and classification labels. Also compute classification and regression losses.
        """
        
        # Initiate losses:
        self.FRCNN_CLASS_LOSS = 0
        self.FRCNN_BBOX_LOSS  = 0
        
        assert(list(batch.size()) == Config.FRCNN.FEATURE_SIZE)
        assert(batch.size(0) == gt_boxes.size(0))
        
        # Get low-level feature map from the CNN. ________________________________________________________________________
        # NOTE: Since this is an adaptation of the Fast R-CNN network, these
        #       convolution operations are shared by the proposal and
        #       classification network.
        _featureMap = self.VGG16FeatureExtractor(batch)
        
        # Obtain ROI proposals from the Region Proposals Network: ________________________________________________________________________
        proposals, rpn_cls_loss, rpn_pred_loss = \
            self.RPNLayer(_featureMap, gt_boxes)
        
        # If this is training, further refine the proposals through
        # ground-truth bounding boxes.
        _labels, bbox_targets = [None, None]
        if Config.TRAIN.TRAINING:
            proposals, _labels, bbox_targets, bbox_inweights, bbox_outweights =\
                self.PropTargetLayer(proposals, gt_boxes)
            
            # Convert these tensors to autograd variables:
            _labels         = Variable(_labels)
            bbox_targets    = Variable(bbox_targets)
            bbox_inweights  = Variable(bbox_inweights)
            bbox_outweights = Variable(bbox_outweights)
        
        # Perform ROI Max Pooling to create feature sets of the same size for 
        # obtained proposals. ________________________________________________________________________
        # Append batch indices to proposal coordinates and permute them to 
        # [k x1 y1 x2 y2] format as required by roi_pool.
        # poposals - [b N 4] tensor
        _b, _N, _ = proposals.size()
        batchids  = torch.from_numpy(np.repeat(np.arange(_b), _N)).float()
        batchids  = batchids.float().view(-1, 1)
        rois      = Variable(torch.cat((batchids, proposals.view(-1,4)), dim=1))
        
        # Pool the proposals from the batch:
        _featurePool = roi_pool(_featureMap, rois, 
                                Config.FRCNN.ADAPTIVEPOOL_SIZE,
                                Config.FRCNN.SPATIAL_SCALE)
        
        # Feed this pooled features to the R-CNN head:
        _featurePool = _featurePool.view(_featurePool.size(0), -1)
        _featurePool = self.VGG16FeatureClassifier(_featurePool)
        
        # Classify and predict bounding boxes:
        bboxes   = self.FRCNNBBoxPred(_featurePool)
        bboxes   = bboxes.view(batch.size(0), _N, -1)
        bscores  = self.FRCNNClassScore(_featurePool)
        cls_prob = fcn.softmax(bscores, 1)
        cls_prob = cls_prob.view(batch.size(0), _N, -1)
        
        # For training workflows, compute use the computed targets and the
        # labels to compute total loss.
        if Config.TRAIN.TRAINING:
            self.FRCNN_CLASS_LOSS = fcn.cross_entropy(bscores, 
                                                      _labels.view(-1).long())

            # Bounding box regression L1 loss
            self.FRCNN_BBOX_LOSS  = smoothL1Loss(bboxes, 
                                                 bbox_targets, 
                                                 bbox_inweights,
                                                 bbox_outweights)

        
        # DONE: Return all variables required by the trainer to identify
        #       whether the weights have converged.
        return  proposals,             \
                bboxes,                \
                cls_prob,              \
                rpn_pred_loss,         \
                rpn_cls_loss,          \
                self.FRCNN_BBOX_LOSS,  \
                self.FRCNN_CLASS_LOSS, \
                _labels
#_______________________________________________________________________________