"""
Module Configuration Class.

Contains constant configuration hyperparameters for BigTIFF, RPN, Faster R-CNN
modules. To access these configuration paramters, statically call this class
attributes.

    FRCNN - Faster R-CNN Network
"""

VGG_SPATIAL_SCALE = 16

class Config:
    """
        Static container class that contains all config parameters.
    """

    # Training Parameters ____________________________________________________________________________
    TRAIN         = True
    NUM_WORKERS   = 0
    LEARNING_RATE = 0.001
    MOMENTUM      = 0.9
    WEIGHT_DECAY  = 0.0005
    GAMMA         = 0.1
    LR_STEPSIZE   = 10000
    MAX_EPOCH     = 20
    DISPLAY_STEP  = 10
    
    # Faster R-CNN Parameters ____________________________________________________________________________
    class FRCNN:
        FEATURE_SIZE      = [4, 3, 256, 256]
        SPATIAL_SCALE     = VGG_SPATIAL_SCALE
        ADAPTIVEPOOL_SIZE = (7, 7)
        ROI_POSITIVE_THR  = (0.5, 1.0)
        ROI_NEGATIVE_THR  = (0.0, 0.5)
        ROI_SAMPLES       = 128
        ROI_SAMPLE_RATIO  = 0.25
        MAX_GT_BOXES      = 20
    
    # RPN Parameters ____________________________________________________________________________
    class RPN:
        
        # Network Parameters ________________________________________________________________________
        FEATURE_DEPTH = 512
        MIN_BOX_SIZE  = 8
        
        # Anchor Parameters ________________________________________________________________________
        ANCHOR_SCALES       = [4, 8, 16]
        ANCHOR_RATIOS       = [0.5, 1, 2]
        SPATIAL_SCALE       = VGG_SPATIAL_SCALE
        ANCHOR_SAMPLES      = 256
        ANCHOR_SAMPLE_RATIO = 0.5
        
        # Intersection-Over-Union Parameters ________________________________________________________________________
        IOU_POSITIVE_THR = (0.7, 1.0)
        IOU_NEGATIVE_THR = (0.0, 0.3)
        
        # Non-Maximal Supression Parameters ________________________________________________________________________
        PRE_TRAIN_NMS_N  = 12000
        POST_TRAIN_NMS_N = 2000
        PRE_TEST_NMS_N   = 6000
        POST_TEST_NMS_N  = 300
        NMS_THR          = 0.7
        
        # Smooth L1 Loss Function Parameters ________________________________________________________________________
        L1WEIGHT_INTERSECTION = 1
        L1WEIGHT_DIFFERENCE   = 1

    