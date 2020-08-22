import torch

class Config:
    
    # All network parameters go here:
    DEVICE        = torch.device('cpu')
    NUM_WORKERS   = 4
    LR_STEPSIZE   = 0.3
    BATCH_SIZE    = 4
    IMAGE_SIZE    = [800, 800, 3]
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 5e-4
    GAMMA         = 0.1
    LR_STEPSIZE   = 10000
    MAX_EPOCH     = 20
    DISPLAY_STEP  = 10
    NUM_ROI       = 20
    NMS_THR       = 0.7
    MIN_BOX_SIZE  = 16