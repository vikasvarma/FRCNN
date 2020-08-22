"""
frcnn - Faster R-CNN Object Detection Network Module
"""

from   torch           import nn
from   config          import Config
from   .utils          import *
from   torchvision.ops import roi_pool
from   torchvision     import transforms
from   PIL             import Image
import torch
import torchvision
import torch.nn.functional as F

"""Random Normal initialization of a NN module layer"""
def normal_init(module, mean, std):
    module.weight.data.normal_(mean, std)
    module.bias.data.zero_()

"""Macro to evaluate class functional attribute without gradient calculations"""
def nograd(fcn):
    def new_fcn(*args,**kwargs):
        with torch.no_grad():
           return fcn(*args,**kwargs)
    return new_fcn

#-------------------------------------------------------------------------------
#  Faster R-CNN Object Detection Network
#-------------------------------------------------------------------------------
class FRCNN(nn.Module):
    """
    """
    spatial_scale = 16
    
    #---------------------------------------------------------------------------
    def __init__(self, classes, image_size, rpn=None):
        # Create the components of the Faster R-CNN network.
        super(FRCNN, self).__init__()
        
        self.image_size = image_size
        self.classes    = classes
        
        # Create Faster R-CNN Network Layers: 
        #_______________________________________________________________________
        # Extract the pretrained convolutional layers from a VGG-16 network
        # which will be used as the base feature extractor.
        vgg        = torchvision.models.vgg16(pretrained=True)
        extractor  = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.VGG16FeatureExtractor  = extractor
        
        # Fix the first 10 layers of VGG, and don't train over them:
        for module in range(10):
            for param in self.VGG16FeatureExtractor[module].parameters(): 
                param.requires_grad = False
        
        # Create the RPN network layer which provides region proposals from
        # computed convolutional features:
        if rpn is None:
            self.RPN = RPN(image_size, stride = self.spatial_scale)
        else:
            self.RPN = rpn
        
        # Create the head:
        self.Head = RCNNHead(
            classifier    = classifier,
            num_classes   = len(classes) + 1,
            spatial_scale = self.spatial_scale
        )
    
    #---------------------------------------------------------------------------
    def __initialize__(self, mean, std):
        # Initialize the network components with random-normal weights:
        self.RPN.__initialize__(mean, std)
        self.Head.__initialize__(mean, std)
    
    #---------------------------------------------------------------------------
    def __set_numproposals__(self, pre_nms, post_nms):
        self.RPN.num_post_nms = post_nms
        self.RPN.num_pre_nms  = pre_nms
    
    #---------------------------------------------------------------------------
    def forward(self, batch):
        """ Feed forward the batch of images through the network to predict     
            object bounding boxes and classification labels.
        """
        
        assert(list(batch.size(0)) == Config.BATCH_SIZE)
        assert(list(batch.size(2)) == self.image_size(0))
        assert(list(batch.size(3)) == self.image_size(1))
        
        # Get low-level feature map from the CNN. 
        #_______________________________________________________________________
        # NOTE: Since this is an adaptation of the Fast R-CNN network, these
        #       convolution operations are shared by the proposal and
        #       classification network.
        features = self.VGG16FeatureExtractor(batch)
        
        # Obtain ROI proposals from the Region Proposals Network: 
        #_______________________________________________________________________
        proposals, scores, targets = self.RPN(features)
        
        # Pass on the targets from the RPN to the head: 
        #_______________________________________________________________________
        roi_targets, roi_scores = self.Head(features, proposals)
        
        return roi_targets, roi_scores, proposals
    
    #---------------------------------------------------------------------------
    def __nms__(self, roi_box, roi_prob):
        """ Apply non-maximal suppression of predicted ROI boxes"""
        
        # Define the threshold for score: Only ROIs which pass this lower bound 
        # probability will be selected for supression.
        thr = 0.5
        
        box, label, score = list()
        
        # Skip the 0th class, which is the background class:
        roi_box = roi_box.reshape((-1, len(self.classes + 1), 4))
        for labelid in range(1, len(self.classes) + 1):
            label_box  = roi_box[:, labelid, :]
            label_prob = roi_prob[:, labelid]
            
            label_box  = label_box[ label_prob > thr ]
            label_prob = label_prob[ label_prob > thr ]
            
            keep = nms(label_box, label_prob)
            
            # Labels returned are in [0, NUM_CLASSES - 2].
            bbox.append(label_box[keep])
            label.append((labelid - 1) * torch.ones((len(keep),)))
            score.append(label_prob[keep])
            
        bbox  = torch.cat(bbox , dim=0).to(torch.float32)
        label = torch.cat(label, dim=0).to(torch.float32)
        score = torch.cat(score, dim=0).to(torch.float32)
        
        return bbox, label, score
    
    #---------------------------------------------------------------------------
    @nograd
    def predict(self, batch):
        """ Detect objects using a trained Faster R-CNN network.
        """
        
        # Set module to evaluation mode:
        self.eval()
        input_size = batch.size()
        
        # Preprocess images to rescale and normalize them:
        tform = transforms.Resize(self.image_size)
        for b in range(batch.size(0)):
            # Resize image to network image size:
            image = batch[b].permute(1,2,0).numpy()
            image = tform.__call__(Image.fromarray(image))
            image = torch.from_numpy(np.array(image)).permute(2,0,1)
            image = image.contiguous()
            
            # Normalize image intensity to [-1 1]:
            for c in range(image.size(0)):
                image[c] = (image[c] - 128) / 128

            # Reset the batch:
            batch[b] = image
        
        # Scale to convert predicted ROI coordinates to image coordinates:
        # NOTE: Assuming square image sizes
        scale = self.image_size[0] / input_size(2)
        
        # Forward pass:
        roi_targets, roi_scores, proposals = self.forward(batch)
        
        # Rescale proposals & targets and obtain to bounding box coordinates:
        proposals = proposals / scale
        yxroi     = targets2box(roi_targets, proposals)

        # Convert ROI scores to probabilities:
        roi_prob  = F.softmax(roi_scores, dim=1)
        
        # Clamp the ROI coordinates to the image extents:
        yxroi[:,:,0].clamp_(min=0, max=input_size[2]-1)
        yxroi[:,:,1].clamp_(min=0, max=input_size[3]-1)
        yxroi[:,:,2].clamp_(min=0, max=input_size[2]-1)
        yxroi[:,:,3].clamp_(min=0, max=input_size[3]-1)
        
        # Apply non-maximal suppression to narrow down the ROIs from proposals:
        bbox, label, score = self.__nms__(yxroi, roi_prob)
        
        return bbox, label, score
#_______________________________________________________________________________

#-------------------------------------------------------------------------------
#  Faster R-CNN ROI Head Module
#-------------------------------------------------------------------------------
class RCNNHead(nn.Module):
    """R-CNN Head Module to predict and classify ROIs from proposals"""
    
    #---------------------------------------------------------------------------
    def __init__(self, 
            classifier, 
            num_classes, 
            spatial_scale,
            pool_size = (7, 7)
        ):
        super(RCNNHead, self).__init__()
        
        # Create R-CNN Head Layers: 
        #_______________________________________________________________________
        self.Classifier = classifier
        self.RCNNClass  = nn.Linear(4096, num_classes)
        self.RCNNBBox   = nn.Linear(4096, 4)
        self.pool_size  = pool_size
        self.spatial_scale = spatial_scale
      
    #---------------------------------------------------------------------------
    def __initialize__(self, mean, std):
        # Initialize the network components with random-normal weights:
        normal_init(self.RCNNClass, mean, std)
        normal_init(self.RCNNBBox , mean, std)
     
    #---------------------------------------------------------------------------
    def forward(self, batch, proposals):
        """ Feed forward the proposal regions into the RCNN head to predict     
            object ROIs and corresponding classes.
        """
        
        # Perform ROI Max Pooling to create feature sets of the same size for 
        # obtained proposals. 
        #_______________________________________________________________________
        # Append batch indices to proposal coordinates and permute them to 
        # [k x1 y1 x2 y2] format as required by roi_pool:
        B, N, _  = proposals.size()
        batchids = torch.from_numpy(np.repeat(np.arange(B), N))
        batchids = batchids.view(-1,1).to(proposals.device).to(proposals.dtype)
        rois     = torch.cat((batchids, proposals.view(-1, 4)),dim=1)
        
        # NOTE: IMPORTANT - Have to convert YX -> XY
        xyROIs   = rois[:, [0,2,1,4,3]]
        
        # Perform pooling:
        scale    = 1 / float(self.spatial_scale)
        pool     = roi_pool(batch, xyROIs, self.pool_size, spatial_scale=scale)
        
        # Feed pooled features to RCNN head, obtain ROI targets and scores: 
        #_______________________________________________________________________
        pool_features = self.Classifier(pool.view(pool.size(0), -1))
        roi_targets   = self.RCNNBBox(pool_features)
        roi_scores    = self.RCNNClass(pool_features)
        
        # Resize the predictions to batch specific scores:
        #_______________________________________________________________________
        roi_targets   = roi_targets.view([B, N, roi_targets.size(1)])
        roi_scores    = roi_scores.view([B, N, roi_scores.size(1)])
        
        return roi_targets, roi_scores
#_______________________________________________________________________________

#-------------------------------------------------------------------------------
#  Region Proposal Network
#-------------------------------------------------------------------------------
class RPN(nn.Module):
    """
        Region Proposal Network: Class which contains utility methods to compute the region proposals of an image.
    """
    
    #---------------------------------------------------------------------------
    def __init__(
        self,
        image_size,
        feature_depth = 512, 
        rpn_depth = 512,
        anchor_ratios = [0.5, 1, 2], 
        anchor_scales = [8, 16, 32],
        stride = 16
    ):
        super(RPN, self).__init__()
        self.image_size   = image_size
        self.stride       = stride
        self.num_anchor   = len(anchor_ratios) * len(anchor_scales)
        self.num_pre_nms  = 6000
        self.num_post_nms = 300 
        
        # Create Network Layers: 
        #_______________________________________________________________________
        self.ConvLayer  = nn.Conv2d(feature_depth, 
                                    rpn_depth, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1)
        
        self.BBoxLayer  = nn.Conv2d(rpn_depth, 
                                    self.num_anchor * 4, 
                                    kernel_size=1, 
                                    stride=1, 
                                    padding=0)
        
        self.ClassLayer = nn.Conv2d(rpn_depth, 
                                    self.num_anchor * 2, 
                                    kernel_size=1, 
                                    stride=1, 
                                    padding=0)
        
        # Create anchors:
        self.Anchors, self.Inside = self.__generate_anchors__(
                                        stride,
                                        image_size,
                                        anchor_ratios,
                                        anchor_scales
                                    )
    
    #---------------------------------------------------------------------------
    def __initialize__(self, mean, std):
        # Initialize all layers with weights drawn from a Gaussian Distribution 
        # of mean MEAN and standard deviation STD.
        normal_init(self.ConvLayer , mean, std)
        normal_init(self.BBoxLayer , mean, std)
        normal_init(self.ClassLayer, mean, std)
    
    #---------------------------------------------------------------------------
    @staticmethod
    def __generate_anchors__(stride, imsize, anchor_ratios, anchor_scales):
        """
            All images are down-sampled by STRIDE (s) when passed through VGG network, so while creating bounding boxes for region proposals on the feature map, the box should be creating on the input image coordinates at the centre locations of s-by-s image pixels which map to each pixel in the feature map.
        """

        # Create the anchor centers for each feature map location:
        yloc = np.arange(stride/2, imsize[0], stride).astype(int)
        xloc = np.arange(stride/2, imsize[1], stride).astype(int)
        ctrs = np.array(np.meshgrid(xloc, yloc)).T.reshape(-1,2)

        # Create anchors for each centre:
        anchor_size = [[stride * s * np.sqrt(r), stride * s * np.sqrt(1./r)]
                        for s in anchor_scales for r in anchor_ratios]

        anchors = np.empty((0,4), dtype=np.float32)
        for _dim in anchor_size:
            anchors = np.append(
                        anchors, 
                        np.append(ctrs - np.multiply(0.5, _dim),
                                  ctrs + np.multiply(0.5, _dim), axis=1),
                        axis=0
                    )

        # Convert anchors to torch tensors:
        anchors      = torch.from_numpy(anchors).to(torch.float32)
        ctrhwAnchors = centrehw(anchors)
        
        # For training, Faster RCNN only takes into account the set of valid
        # anchors that lie completely within the image, therefore, bound these
        # anchors to get the valid subset.
        keep = (
            (anchors[:,0] >= 0) &
            (anchors[:,1] >= 0) &
            (anchors[:,2] <= imsize[0] - 1) &
            (anchors[:,3] <= imsize[1] - 1)
        )
        inside = torch.nonzero(keep).view(-1)
        
        return anchors, inside
    
    #---------------------------------------------------------------------------
    def __get_anchors__(self):
        return self.Anchors, self.Inside
    
    #---------------------------------------------------------------------------
    def forward(self, features):
        """
        TODO
        
        Forward propagation fed into by the VGG feature extractor.
        
        """
        assert not self.Anchors is None
        
        # Pass inputs through the RPN network and compute proposal targets: 
        #_______________________________________________________________________
        B, C, H, W = features.size()
        
        # Extract RPN feature activations:
        rpn_features   = F.relu(self.ConvLayer(features), inplace=True)
        
        # Derive objectness score and bounding box proposal prediction targets:
        objectness_score = self.ClassLayer(rpn_features)
        objectness_score = objectness_score.permute(0,2,3,1).contiguous()
        objectness_score = objectness_score.view(B, H, W, self.num_anchor, 2)
        
        objectness_prob  = F.softmax(objectness_score, dim=4)
        objectness_prob  = objectness_prob[:,:,:,:,1].contiguous().view(B, -1)
        objectness_score = objectness_score.view(B, -1, 2)
        
        # Compute anchor proposal targets:
        targets = self.BBoxLayer(rpn_features)
        targets = targets.permute(0,2,3,1).contiguous().view(B, -1, 4)
        
        # Clamp the anchors and predicted boxes to image extents: 
        #_______________________________________________________________________
        # Get anchor inside the image:
        anchors = self.Anchors
        anchors = anchors.expand(B, anchors.size(0), 4)
        anchors = anchors.to(targets.device)
        
        # Convert proposals from target coefficients to box coordinates:
        yxBbox  = targets2box(targets, anchors)
        
        # Clamp the proposals to the image extents:
        yxBbox[:,:,0].clamp_(min=0, max=self.image_size[0]-1)
        yxBbox[:,:,1].clamp_(min=0, max=self.image_size[1]-1)
        yxBbox[:,:,2].clamp_(min=0, max=self.image_size[0]-1)
        yxBbox[:,:,3].clamp_(min=0, max=self.image_size[1]-1)
        
        # Apply non-maximal supression to obtain top scoring proposals: 
        #_______________________________________________________________________
        top_proposals = nms(yxBbox, objectness_prob, 
                            self.num_pre_nms, self.num_post_nms)
        top_proposals = torch.floor(top_proposals)
        
        return top_proposals, objectness_score, targets
#_______________________________________________________________________________
