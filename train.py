import numpy as np
import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch            import nn
from torchvision      import transforms
from model.dataset    import STACCarribeanDataset
from trainer          import FRCNNTrainer
from model.transforms import *
from config           import Config

## Setup dataset for training:
#  Define data transforms to introduce randomness into the dataset.

# Set rand seed for reproducibility
np.random.seed(59)

# Prepare dataset:
regions   = "colombia"
tform = transforms.Compose([
    RandomVerticalFlip(),
    RandomCrop(1024-64),
    JitterColor(
        brightness = [0.95, 1.05], 
        contrast   = [0.95, 1.05], 
        saturation = [0.95, 1.05]
    ),
    RandomHorizontalFlip(),
    ImageResize(800),
    ToTensor(device=Config.DEVICE),
    NormalizeIntensity(128, 128)
])

# Prepare dataset with specified transforms and balance the dataset:
trainset = STACCarribeanDataset(regions, train=True , transform=tform)
trainset.balance_classes(np.arange(4))

# Create trainer and start training
if __name__ == '__main__':
    trainer = FRCNNTrainer(trainset, 'logs/stac-openai/')
    for epoch in range(Config.MAX_EPOCH):
        max_steps = int(trainset.__len__() / Config.BATCH_SIZE)
        for step in range(max_steps):
            trainer.dostep(epoch, step)