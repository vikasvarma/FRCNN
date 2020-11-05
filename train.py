import numpy as np
import time
import os
import csv
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch            import nn
from torchvision      import transforms
from model.dataset    import STACCarribeanDataset
from trainer          import FRCNNTrainer
from model.transforms import *
from config           import Config
from torch.utils.data import DataLoader
from model.dataset    import partition

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
        
    # Done training:
    print("Done training. Wohoo!")
    
    # Now, validate the network perdictions on test set:
    network = trainer.network
    num_batches = int(testset.__len__() / Config.BATCH_SIZE)
    
    # Create a dataloader:
    test_loader = DataLoader(
        testset,
        batch_size = Config.BATCH_SIZE,
        shuffle = False,
        num_workers = Config.NUM_WORKERS,
        collate_fn = testset.collate,
        worker_init_fn = partition
    )
    test_itr = iter(test_loader)
    results  = []
    
    for ind in range(num_batches):
        # Load next batch:
        batch  = next(test_itr)
        images = batch[0]
        gt_roi = batch[1].view(-1,4)
        img_id = batch[2]
        
        # Run detection on the batch of images:
        bbox, label, scores = network.predict(images)
        
        # Checkpoint: save bboxes, labels and cooresponding scores:
        patch_ids = img_id[bbox[:,0].to(torch.long)].to(torch.long)
        patch_org = testset.Samples[patch_ids]
        region_id = patch_org[:,0]
        patch_org = patch_org[:,1:]
        roi_cord  = bbox[:,1:].detach().numpy() + np.tile(patch_org, 2) - 1
        roi_cord  = roi_cord.astype(np.long)
        
        for n in range(bbox.size(0)):
            entry = {}
            entry['PATCH_ID'] = patch_ids[n].tolist()
            entry['REGION_ID'] = region_id[n].tolist()
            entry['PATCH_ORIGIN'] = patch_org[n].tolist()
            entry['ROI'] = roi_cord[n].tolist()
            entry['LABEL'] = label[n].tolist()
            entry['SCORE'] = scores[n].tolist()
            results.append(entry)
            
# Save results to csv:
fields = ['PATCH_ID', 'REGION_ID', 'PATCH_ORIGIN', 'ROI', 'LABEL', 'SCORE'] 
with open('test_results.csv', 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = fields) 
    writer.writeheader() 
    writer.writerows(results)  