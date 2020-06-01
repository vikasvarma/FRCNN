# Import required API: ________________________________________________________________________________
import numpy as np
import torch

import torch.nn.functional as F
import torch.optim as optim

from torch.autograd   import Variable
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.dataset    import STACCarribeanDataset, partition
from model.transforms import *
from network.config   import Config
from network.frcnn    import FRCNN
from torch.utils      import tensorboard
from datetime         import datetime

#-------------------------------------------------------------------------------
# Define which device training should happen on:
# assert (torch.cuda.is_available()), "GPU Device is unavailable."
device = torch.device('cpu')

# Setup datasets for train and test: ________________________________________________________________________________
# Define data transforms to introduce randomness into the dataset:
# regions = ["colombia", "guatemala", "st_lucia"]
regions   = "colombia"
adjust    = [0.95, 1.05]
testform  = transforms.Compose([
                ImageResize(256),
                NormalizeIntensity(128, 128)
            ])
trainform = transforms.Compose([
                RandomVerticalFlip(),
                RandomCrop(1024-64),
                JitterColor(brightness=adjust, 
                            contrast=adjust, 
                            saturation=adjust),
                RandomHorizontalFlip(),
                ImageResize(256),
                ToTensor(device=device),
                NormalizeIntensity(128, 128)
            ])

# Prepare dataset with specified transforms:
trainset = STACCarribeanDataset(regions, train=True , transform=trainform)
testset  = STACCarribeanDataset(regions, train=False, transform=testform)

# Seed RNG to get reproducible results:
rng_state = np.random.get_state()
np.random.seed(59)

# Class balance trainset:
trainset.balance_classes(np.arange(4))

# Wrap around dataloaders with parallel loading:
trainloader  = DataLoader(trainset, 
                          batch_size=Config.FRCNN.FEATURE_SIZE[0], 
                          shuffle=True, 
                          num_workers=Config.NUM_WORKERS, 
                          collate_fn=trainset.collate,
                          worker_init_fn=partition)

testloader   = DataLoader(testset, 
                          batch_size=Config.FRCNN.FEATURE_SIZE[0], 
                          shuffle=False, 
                          num_workers=Config.NUM_WORKERS, 
                          collate_fn=testset.collate,
                          worker_init_fn=partition)

# Get object classes:
classes = trainset._classes

if __name__ == '__main__':
    #---------------------------------------------------------------------------# Setup Model & Network Parameters ____________________________________________________________________________# Initialize the network and move to the right device:
    net = FRCNN(classes)
    net.to(device)

    # Initialize network inputs:
    images = torch.FloatTensor(0).to(device).resize_(Config.FRCNN.FEATURE_SIZE)
    bboxes = torch.FloatTensor(0).to(device).resize_([4, 20, 5])
    
    # Use SGD optimizer for training with learning rate adjusted after steps 
    # of epochs using a learning rate scheduler:
    params    = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params,
                                lr=Config.LEARNING_RATE, 
                                weight_decay=Config.WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                             step_size=Config.LR_STEPSIZE,
                                             gamma=Config.GAMMA)
    
    # Create tensorboard writer to log trainings:
    logger = tensorboard.SummaryWriter("logs/STAC-OpenAI")
    
    # Training -----------------------------------------------------------------
    for epoch in range(Config.MAX_EPOCH):
        # Iterate through all samples in the train set:
        trainsize  = int(trainset.__len__() / Config.FRCNN.FEATURE_SIZE[0])
        trainiter  = iter(trainloader)
        net.train()
        
        # Initialize loss tracker for tensorbard:
        accum_loss = 0
        
        for step in range(trainsize):
            # Clear gradients:
            net.zero_grad()
            
            # Load batch and copy to variables:
            batch  = next(trainiter)
            images.data = batch[0].data
            bboxes.data = batch[1].data
            
            # Forward pass:
            proposals, targets, cls_prob, \
            Lrpn_pred, Lrpn_cls, Lrcnn_pred, Lrcnn_cls, labels = \
            net(images, bboxes)
            
            # Compute total loss:
            loss = Lrpn_pred.mean()  + Lrpn_cls.mean() + \
                   Lrcnn_pred.mean() + Lrcnn_cls.mean()
            
            # Back-propagation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Keep an account of loss:
            accum_loss += loss.item()
            
            if step % Config.DISPLAY_STEP == 0:
                # Time to print some results...
                N = epoch * trainsize + step
                if step > 0: accum_loss /= Config.DISPLAY_STEP
                
                # Divide loss by batch size:
                accum_loss /= trainloader.batch_size
                
                # Print progress:
                stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                info="[%s][TRAIN][Epoch %2d][Step %4d:%4d] Loss: %.4f, LR: %.2e"
                lr  = optimizer.param_groups[0]["lr"]
                print(info % (stamp, epoch, step, trainsize, accum_loss, lr))
                
                # Covert losses from tensors to float:
                Lrpn_cls   = Lrpn_cls.item()
                Lrpn_pred  = Lrpn_pred.item()
                Lrcnn_cls  = Lrcnn_cls.item()
                Lrcnn_pred = Lrcnn_pred.item()
                
                # Log loss to tensorboard:
                logger.add_scalar("LOSS/TRAIN/Cumulative_Loss", accum_loss, N)
                logger.add_scalar("LOSS/TRAIN/RPN_Class_Loss" , Lrpn_cls  , N)
                logger.add_scalar("LOSS/TRAIN/RPN_Reg_Loss"   , Lrpn_pred , N)
                logger.add_scalar("LOSS/TRAIN/RCNN_Class_Loss", Lrcnn_cls , N)
                logger.add_scalar("LOSS/TRAIN/RCNN_Reg_Loss"  , Lrcnn_pred, N)
                
                # Reset accumulated loss:
                accum_loss = 0
            
    # Save Trained Model -------------------------------------------------------
    print("Training Complete. Yayy!")
    FILE = "./stac-openai-carribean-frcnn-vgg16.pth"
    torch.save(net.state_dict(), FILE)
    print("Saved Model: {0}".format(FILE))
        
    # Test Set Accuracy --------------------------------------------------------
    # Load model back:
    net = FRCNN(classes);
    net.load_state_dict(torch.load(FILE))
    print("Load Complete: {0}".format(FILE))
        
    # Configure network for evaluation:
    net.cuda()
    net.eval()
        
    # Evaluate:
    testsize = int(testset.__len__() / Config.FRCNN.FEATURE_SIZE[0])
    testiter = iter(testloader)
        
    # Initialize loss tracker for tensorbard:
    accum_loss = 0
        
    for step in range(testsize):
        # Load batch and copy to variables:
        batch = next(trainiter)
        images.data.resize_(batch[0].size()).copy_(batch[0])
        bboxes.data.resize_(batch[1].size()).copy_(batch[1])
        
        # Forward pass:
        results = net(images, bboxes)
        scores  = results[2].data
        targets = results[1].data[:, 1:5]
            
        # TODO: Evaludation and saving targets to file.
        

    print("That's it!")