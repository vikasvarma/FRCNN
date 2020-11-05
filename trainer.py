# Import required API: ________________________________________________________________________________
import numpy as np
import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import datetime

from torch            import nn
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.bigtiff    import BigtiffDataset
from model.transforms import *
from model.utils      import *
from model.dataset    import partition
from model.frcnn      import FRCNN
from torch.utils      import tensorboard
from config           import Config
from collections      import namedtuple
from utils            import batchplot

#-------------------------------------------------------------------------------
#   Class definition macro for modules with no back-propagation gradient 
#   calculation.
#-------------------------------------------------------------------------------
def nograd(fcn):
    def no_grad_fcn(*args, **kwargs):
        with torch.no_grad():
           return fcn(*args, **kwargs)
    return no_grad_fcn

Loss = namedtuple('Loss', [
    'rpn_box_loss' ,
    'rpn_cls_loss' ,
    'rcnn_box_loss',
    'rcnn_cls_loss',
    'total_loss'   
])

#-------------------------------------------------------------------------------
#   AnchorTargetGenerator - Generates anchor targets wrt ground truth boxes
#-------------------------------------------------------------------------------
class AnchorTargetGenerator(object):
    
    num_samples = 256
    pos_iou_thr = 0.7
    neg_iou_thr = 0.3
    ratio       = 0.5
    
    @staticmethod
    def __call__(anchors, gt_bbox, inside):
        """ Assign ground truth bounding boxes to anchors for training the 
            Region Proposal Network in Faster R-CNN.
        """
        
        B = gt_bbox.size(0)
        if anchors.dim() == 2:
            anchors = anchors.expand(B,anchors.size(0),4).contiguous()
                
        # Compute ground truth boxes overlap with anchors and label them:
        iou = IoU(gt_bbox, anchors)
        gt_anchor_label, argmax_bbox = classify(
            iou, 
            AnchorTargetGenerator.pos_iou_thr,
            AnchorTargetGenerator.neg_iou_thr
        )
        
        # Ignore out of bound anchors:
        gt_anchor_label[:, inside] = -1;
        
        # Randomly sample a fixed subset of the anchors:
        gt_anchor_label = randsample(
            gt_anchor_label,
            AnchorTargetGenerator.num_samples,
            AnchorTargetGenerator.ratio
        )
        
        # Calculate anchor targets:
        # NOTE: Select only the ground truth objects which have the maximum
        #       overlap with each anchor to compute the target coefficients.
        boxes = gt_bbox.new(anchors.size()).zero_()
        for b in range(B):
            boxes[b] = gt_bbox[b, argmax_bbox[b], :]
        
        # Map computed targets to only contain inside boxes:
        targets = box2targets(anchors, boxes)
        gt_anchor_targets = gt_bbox.new(anchors.size()).zero_()
        gt_anchor_targets[:, inside, :] = targets[:, inside, :]
        
        return gt_anchor_targets, gt_anchor_label
#_______________________________________________________________________________

#-------------------------------------------------------------------------------
#   ProposalTargetGenerator - Generates proposal targets wrt ground truth boxes
#-------------------------------------------------------------------------------
class ProposalTargetGenerator(object):
    
    num_samples = 128
    pos_iou_thr = 0.5
    neg_iou_thr = 0.5
    ratio       = 0.25
    
    @staticmethod
    def __call__(proposals, gt_bbox, gt_label):
        """ Assign ground truth targets to sampled proposals.
        """
        
        B, N, _ = gt_bbox.size()
        S = ProposalTargetGenerator.num_samples
        
        # Concatenate proposals and ground truth boxes:
        proposals = torch.cat((proposals, gt_bbox), dim=1)
        iou       = IoU(gt_bbox, proposals)
        proposal_label, argmax_bbox = classify(
            iou, 
            ProposalTargetGenerator.pos_iou_thr,
            ProposalTargetGenerator.neg_iou_thr
        )
        
        # Randomly sample a fixed subset of the anchors:
        proposal_label = randsample(
            proposal_label,
            ProposalTargetGenerator.num_samples,
            ProposalTargetGenerator.ratio
        )
        
        # Select sampled proposals:
        keep = proposal_label >= 0
        sample_rois  = proposals.new(torch.Size((B, S, 4)))
        gt_roi_label = gt_label.new(torch.Size((B, S)))
        gt_assigned  = gt_bbox.new(torch.Size((B, S, 4)))
        
        # Get sampled rois and labels:
        for b in range(B):
            # NOTE: Offset the labels to account for 0 labeled background.
            gt_ind = argmax_bbox[b]
            labels = gt_label[b, gt_ind[keep[b]]]
            sample_rois[b]  = proposals[b, keep[b], :]
            gt_roi_label[b] = labels
            gt_assigned[b]  = gt_bbox[b, gt_ind[keep[b]]]

        # Compute sampled ROI targets wrt ground truth:
        gt_roi_targets = box2targets(sample_rois, gt_assigned)
        
        # Offset ROI lagels from [0 L-1] to [1 L] to allow for 0 to be 
        # background class label.
        gt_roi_label = gt_roi_label + 1
        
        # NOTE: TO REMOVE
        sample_label = proposal_label[keep].view([B,S])
        
        return sample_rois, gt_roi_targets, gt_roi_label, sample_label
#_______________________________________________________________________________

#-------------------------------------------------------------------------------
#   Faster R-CNN Object Detetction Network Trainer for BigTIFF #-------------------------------------------------------------------------------
class FRCNNTrainer(nn.Module):
    """ Faster R-CNN Trainer for training an object detection network with a 
        BigTIFF dataset.
    """
    
    #---------------------------------------------------------------------------
    def __init__(self, dataset, logdir):
        """ Initialize the trainer with a training dataset.
        """
        assert isinstance(dataset, BigtiffDataset)
        
        super(FRCNNTrainer, self).__init__()
        
        self.dataset     = dataset
        self.logdir      = logdir
        self.num_classes = len(dataset.get_classes())
        self.network     = FRCNN(dataset.get_classes(), Config.IMAGE_SIZE)
        self.optimizer   = self.__getoptimizer__()
        self.network.__initialize__(0, 0.01)
        
        # Define number of proposals generated by the RPN network:
        self.network.__set_numproposals__(12000, 2000)
        
        # Create tensorboard writer to log trainings:
        self.logger = tensorboard.SummaryWriter(logdir)
        
        # Setup a learning rate scheduler:
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size = Config.LR_STEPSIZE,
            gamma = Config.GAMMA
        )
        
        # Create a dataloader:
        self.loader = DataLoader(
            self.dataset,
            batch_size = Config.BATCH_SIZE,
            shuffle = True ,
            num_workers = Config.NUM_WORKERS,
            collate_fn = self.dataset.collate,
            worker_init_fn = partition
        )
        self.iterator = None
        
        # Saved network path:
        self.saved_network = None
        
        # Initialize network inputs:
        image_size  = ([Config.BATCH_SIZE] + Config.IMAGE_SIZE)
        roi_size    = (Config.BATCH_SIZE,Config.NUM_ROI,4)
        label_size  = (Config.BATCH_SIZE,Config.NUM_ROI)
        device      = torch.device(Config.DEVICE)
        
        self.batch  = torch.Tensor((image_size)).to(torch.float64)
        self.boxes  = torch.Tensor((roi_size)).to(torch.float64)
        self.labels = torch.Tensor((label_size)).to(torch.float64)
        
        self.batch  = self.batch.to(Config.DEVICE).requires_grad_()
        self.boxes  = self.boxes.to(Config.DEVICE).requires_grad_()
        self.labels = self.labels.to(Config.DEVICE).requires_grad_()
        
    #---------------------------------------------------------------------------
    @staticmethod
    def __targets_loss__(targets, gt_targets, gt_labels, sigma=3):
        
        # Compute weights:
        B, N = gt_labels.size()
        weight = torch.zeros(gt_labels.shape, dtype=torch.float32)
        weight.masked_fill_(gt_labels > 0, 1)
        weight = weight.view(B, N, 1).expand_as(gt_targets)
        
        loss = smoothL1Loss(targets, gt_targets, weight.detach(), sigma)
        
        # IMPORTANT: Normalize by total number of sampled ROIs.
        loss /= ((gt_labels >= 0).sum().float()) 
        
        return loss
    
    #---------------------------------------------------------------------------
    @staticmethod
    def __class_loss__(cls_score, gt_label):
        
        # Compute batch loss:
        B, N, _ = cls_score.size()
        loss = 0
        for b in torch.arange(B):
            batch_score = cls_score[b]
            batch_label = gt_label[b].type(torch.LongTensor)
            loss += F.cross_entropy(batch_score, batch_label, ignore_index = -1)
            
        # Compute mean batch loss:
        loss = loss / B
        
        return loss
    
    #---------------------------------------------------------------------------
    def __getoptimizer__(self):
        """ Get optimizer for training"""
        lr = Config.LEARNING_RATE
        wd = Config.WEIGHT_DECAY
        params = []
        
        for key, value in dict(self.network.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr':lr, 'weight_decay': wd}]
                  
        optimizer = torch.optim.Adam(params)
        return optimizer
    
    #---------------------------------------------------------------------------
    def forward(self, batch, gt_bbox, gt_label):
        """Forward propagate the trainer and compute prediction losses"""
        
        B, C, H, W = batch.size()
        
        # Forward pass through the network, manually:
        #_______________________________________________________________________
        # Get low-level features from the convolution layers:
        features = self.network.VGG16FeatureExtractor(batch)
        
        # Get anchor proposals and targets from RPN:
        anchors, inside = self.network.RPN.__get_anchors__()
        proposals, rpn_score, rpn_targets = self.network.RPN(features)

        # Sample proposals and find ground truth targets:
        sample_proposals, gt_bbox_targets, gt_bbox_label, sample_label=         ProposalTargetGenerator.__call__(
            proposals,
            gt_bbox,
            gt_label
        )
        
        # Get roi bounding boxes and classify using the RCNN head:
        roi_targets, roi_scores = self.network.Head(features, sample_proposals) 
        
        gt_anchor_targets, gt_anchor_label = \
        AnchorTargetGenerator.__call__(
            anchors,
            gt_bbox,
            inside
        )
        
        # Compute losses: 
        #_______________________________________________________________________
        rpn_box_loss  = self.__targets_loss__(
            rpn_targets,
            gt_anchor_targets,
            gt_anchor_label,
            sigma = 3
        )
        
        rpn_cls_loss  = self.__class_loss__(rpn_score, gt_anchor_label)
        
        rcnn_box_loss = self.__targets_loss__(
            roi_targets,
            gt_bbox_targets,
            gt_bbox_label,
            sigma = 1
        )
        
        rcnn_cls_loss = self.__class_loss__(roi_scores, gt_bbox_label)
        
        losses = [rpn_box_loss, rpn_cls_loss, rcnn_box_loss, rcnn_cls_loss]
        losses = losses + [sum(losses)]
        
        return Loss(*losses)
    
    #---------------------------------------------------------------------------
    def dostep(self, epoch, step):
        if step == 0:
            self.iterator = iter(self.loader)
            if epoch > 0:
                # Save the network and optimizer:
                self.save()
        
        # Load batch and copy to variables:
        batch_data = next(self.iterator)
        img_data   = batch_data[0]
        roi_data   = batch_data[1]
        roi_locs   = roi_data[:,:,:4]
        roi_label  = torch.squeeze(roi_data[:,:,4])
        
        # Copy data to the variables:
        self.batch.data  = img_data.data
        self.boxes.data  = roi_locs.data
        self.labels.data = roi_label.data
        
        # Clear previous step gradients:
        self.optimizer.zero_grad()
        
        loss = self.forward(self.batch, self.boxes, self.labels)
        loss.total_loss.backward()
        
        self.optimizer.step()
        if epoch > 0 & step == 0 & epoch % Config.LR_STEPSIZE == 0:
            self.lr_scheduler.step()
        
        if step % Config.DISPLAY_STEP == 0:
            self.log_results(epoch, step, loss)
    
    #---------------------------------------------------------------------------
    def log_results(self, epoch, step, loss):
        # Time to print some results...
        trainsize = int(self.dataset.__len__() / Config.BATCH_SIZE)
        N = epoch * trainsize + step
             
        # Covert losses from tensors to float:
        rpn_box_loss  = loss.rpn_box_loss.item()
        rpn_cls_loss  = loss.rpn_cls_loss.item()
        rcnn_box_loss = loss.rcnn_box_loss.item()
        rcnn_cls_loss = loss.rcnn_cls_loss.item()
        total_loss    = loss.total_loss.item()
        
        # Print progress:
        stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info  ="[%s][TRAIN][Epoch %2d][Step %4d:%4d] Loss: %.4f, LR: %.2e"
        lr    = self.optimizer.param_groups[0]["lr"]
        print(info % (stamp, epoch, step, trainsize, total_loss, lr))
        
        # Log loss to tensorboard:
        self.logger.add_scalar("LOSS/TRAIN/Total_Loss"     , total_loss   , N)
        self.logger.add_scalar("LOSS/TRAIN/RPN_Class_Loss" , rpn_cls_loss , N)
        self.logger.add_scalar("LOSS/TRAIN/RPN_BBox_Loss"  , rpn_box_loss , N)
        self.logger.add_scalar("LOSS/TRAIN/RCNN_Class_Loss", rcnn_cls_loss, N)
        self.logger.add_scalar("LOSS/TRAIN/RCNN_BBox_Loss" , rcnn_box_loss, N)
    
    #---------------------------------------------------------------------------
    def save(self, path=None):
        save_dict              = dict()
        save_dict['model']     = self.network.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        if path is None:
            path = 'checkpoint/frcnn_{0}'.format(time.strftime('%m%d%H%M'))
        
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, path)
        self.saved_network = path
    
    #---------------------------------------------------------------------------
    def load(self, path):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.network.load_state_dict(state_dict['model'])
        
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            
        return self   
#_______________________________________________________________________________