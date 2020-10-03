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
# Set rand seed for reproducibility
np.random.seed(0)

# Prepare dataset:
regions = ["colombia", "guatemala", "st_lucia"]

# Define data transforms to introduce randomness into the training dataset.
tform   = transforms.Compose([
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

# Pre-processing:
pre_process = transforms.Compose([
    ImageResize(800),
    ToTensor(device=Config.DEVICE),
    NormalizeIntensity(128, 128)
])

# Prepare dataset with specified transforms and balance the dataset:
trainset = STACCarribeanDataset(regions, train=True , transform=tform)
testset  = STACCarribeanDataset(regions, train=False, transform=pre_process)

# Divide samples among train and test datasets:
N_sample = len(trainset.Samples)
samples  = trainset.Samples[torch.randperm(N_sample)]
N_train  = round(0.3 * N_sample)

# Select the first 30% of the samples for training and rest for test:
trainset._set_sampling_scheme_(samples[:N_train])
testset._set_sampling_scheme_(samples[N_train:])

# Class balance training dataset:
trainset.balance_classes(np.arange(4))

# Create trainer and start training
if __name__ == '__main__':
    trainer = FRCNNTrainer(trainset, 'logs/stac-openai/')
    for epoch in range(Config.MAX_EPOCH):
        max_steps = int(trainset.__len__() / Config.BATCH_SIZE)
        for step in range(max_steps):
            trainer.dostep(epoch, step)
        
        # Save network after every epoch:
        trainer.save()
        
    # Done training:
    print("Done training. Wohoo!")