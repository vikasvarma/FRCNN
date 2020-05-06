# Import required API: ________________________________________________________________________________
import numpy as np
import torch

import torch.nn.functional as F
import torch.optim as optim

from torchvision      import transforms
from torch.utils.data import DataLoader
from model.dataset    import STACCarribeanDataset, partition
from model.transforms import *
from network.config   import Config
from network.frcnn    import FRCNN

from trainutils import batchplot

# Setup datasets for train and test: ________________________________________________________________________________
# Construct dataset:
dataset = STACCarribeanDataset("colombia")
regions = list(dataset.BigTIFFs.keys())

# Define data transforms to introduce randomness into the dataset:
adjust  = [0.75, 1.5]
tform   = transforms.Compose([
                RandomVerticalFlip(),
                RandomCrop(Config.FRCNN.FEATURE_SIZE[2]),
                JitterColor(brightness=adjust, 
                            contrast=adjust, 
                            saturation=adjust,
                            hue=[-0.1, 0.1]),
                RandomHorizontalFlip(),
                ToTensor()
            ]) if Config.TRAIN else None

# Prepare dataset with specified transforms:
dataset = STACCarribeanDataset("colombia",train=Config.TRAIN,transform=tform)

# Class balancing: Perform a seeding experiment to find an ideal seed.
dataset.balance_classes(np.arange(4), seed=59)

# Wrap around dataloaders with parallel loading:
loader  = DataLoader(dataset, 
                     batch_size=Config.FRCNN.FEATURE_SIZE[0], 
                     shuffle=Config.TRAIN, 
                     num_workers=Config.NUM_WORKERS, 
                     collate_fn=dataset.collate,
                     worker_init_fn=partition)
#-------------------------------------------------------------------------------
# Setup Model & Network Parameters ________________________________________________________________________________
# Parametesr:
learning_rate = Config.LEARNING_RATE
momentum      = Config.MOMENTUM
weight_decay  = Config.WEIGHT_DECAY

# Initialize the network:
net = FRCNN(dataset._classes)
optimizer = optim.A


if __name__ == '__main__':
    iterator = iter(loader)
    
    for epoch in range(1):
        # Load batch and forward pass:
        batch = next(iterator)
